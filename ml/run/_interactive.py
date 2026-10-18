"""The question-and-answer front ends of train.py and watch.py.

Each wizard first offers to use the saved settings in default.json
(see _defaults.py); otherwise it asks for the settings one at a time (press Enter to accept the
default in [brackets]; a bad answer is explained and asked again). Either way it returns a *plan*:
a dictionary describing what to do, or None if you decline at the final "Start?".

  train_wizard -> {"method": "evolution" | "dqn", "run": name, "reset": bool, "count": generations or
                   rounds, "seed": int | None, "save_every": int, "verbose": bool, "config": Config | DQNConfig}
                  (ml.plans.run_training carries it out)
  watch_wizard -> {"kind": "single" | "multiple" | "vs", ...}   (game_interface.plan.build_watch carries it out)
"""
import argparse
import os

from _args import any_int, gamma_value, non_negative_int, playback_speed, positive_float, positive_int, run_name
from _defaults import TRAIN_KINDS, SettingsError, train_plan_from_file, watch_plan_from_file

NOUN = {"gen": "generation", "round": "round"}   # what a saved checkpoint is called in each kind of run


# --- prompts -----------------------------------------------------------------

def ask(prompt, default, convert=str, *, shown=None, input_fn=input, out=print):
    """Ask until `convert` accepts the answer; Enter returns `default` unchanged."""
    shown = default if shown is None else shown
    while True:
        raw = input_fn(f"{prompt} [{shown}]: ").strip()
        if not raw:
            return default
        try:
            return convert(raw)
        except (argparse.ArgumentTypeError, ValueError) as e:
            out(f"  ! {e}")


def ask_yes_no(prompt, default=False, **io):
    def convert(text):
        if text.lower() in ("y", "yes"):
            return True
        if text.lower() in ("n", "no"):
            return False
        raise ValueError("please answer y or n")

    return ask(prompt, default, convert, shown="Y/n" if default else "N/y", **io)


def ask_choice(title, labels, default=1, *, input_fn=input, out=print):
    """Numbered menu; returns the chosen number, counting from 1."""
    out(title)
    for i, label in enumerate(labels, 1):
        out(f"  {i}) {label}")

    def convert(text):
        if not text.isdigit() or not 1 <= int(text) <= len(labels):
            raise ValueError(f"enter a number from 1 to {len(labels)}")
        return int(text)

    return ask("Choose", default, convert, input_fn=input_fn, out=out)


def _confirm(summary, io, warning=None):
    """Show what is about to happen and ask for the go-ahead. summary is a list of (label, value)."""
    out = io["out"]
    width = max(len(label) for label, _ in summary)
    out("\nYour settings:")
    for label, value in summary:
        out(f"  {label:<{width}}  {value}")
    if warning:
        out(f"  ! {warning}")
    return ask_yes_no("Start?", True, **io)


def _list_runs(runs_dir):
    if not os.path.isdir(runs_dir):
        return []
    return sorted(n for n in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, n)))


def _use_settings_file(defaults_path, io, saved_label="Use the saved settings", default=1):
    """Offer the saved settings file. True means the user chose it."""
    if defaults_path is None:
        return False
    choice = ask_choice("How do you want to set this up?",
                        ["Answer the questions", f"{saved_label} in {os.path.basename(defaults_path)}"], default, **io)
    return choice == 2


# --- training ----------------------------------------------------------------

def train_wizard(runs_dir, *, input_fn=input, out=print, defaults_path=None):
    """Ask the kind of training, then either read its settings from the settings file or ask the questions."""
    io = {"input_fn": input_fn, "out": out}
    out("Train a player. Press Enter to accept the value in [brackets].\n")
    method = ask_choice("What kind of training?", [
        "Neuroevolution: a population of networks plays, the best are kept and bred (slower: a minute or two for 100 generations)",
        "Reinforcement learning (DQN): one network learns while it plays (about 30 seconds for 100 rounds)",
    ], 1, **io)
    kind = "evolution" if method == 1 else "dqn"
    out("")
    if _use_settings_file(defaults_path, io, f"Use the saved {TRAIN_KINDS[kind]} settings", default=2):
        try:
            plan, info, fixed = train_plan_from_file(defaults_path, runs_dir, kind)
        except SettingsError as e:
            out(f"\n  ! {e}\n  Fix the file, or answer the questions now.")
        else:
            return _confirm_train(plan, info, fixed, io)
    out("")
    return (_evolution_plan if kind == "evolution" else _dqn_plan)(runs_dir, io)


def _pick_run(runs_dir, io, existing, default_name, other_method, other_hint, noun_plural):
    """Ask for a run name. Returns (name, info, reset): info describes an existing run of this method."""
    from ml.runs import run_method

    out = io["out"]
    while True:
        name = ask("Run name (saved under runs/)", default_name, run_name, **io)
        if run_method(os.path.join(runs_dir, name)) == other_method:
            out(f"  ! '{name}' was trained with {other_hint}. Pick another name.")
            continue
        info, reset = existing.get(name), False
        if info:
            trained = info.generation if hasattr(info, "generation") else info.round
            out(f"  '{name}' has already trained {trained} {noun_plural}; training carries on from there.")
            reset = ask_yes_no("  Delete it and start over instead?", True, **io)
            if not reset and info.hidden is None:
                out("  ! This run's network layout can't be read, so it can't be continued. "
                    "Pick another name, or answer y to start over.")
                continue
        return name, info, reset


def _delete_warning(name, info, noun_plural):
    if info is None:
        return f"This will DELETE runs/{name} before training. Answer n below to cancel."
    trained = info.generation if hasattr(info, "generation") else info.round
    return f"This will DELETE runs/{name} ({trained} {noun_plural} trained) before training. Answer n below to cancel."


def _confirm_train(plan, info, fixed, io):
    """Summarise a training plan and ask for the go-ahead. Returns the plan, or None if declined."""
    cfg, name = plan["config"], plan["run"]
    where = f"runs/{name}" + (" (continuing)" if fixed else " (starting over)" if plan["reset"] else " (new)")
    seed = "from this run" if fixed else str(plan["seed"])
    if plan["method"] == "evolution":
        summary = [("method", "neuroevolution"), ("run", where), ("generations", f"{plan['count']} more"),
                   ("population", f"{cfg.pop_size} individuals, {cfg.elite} elites, {cfg.n_parents} parents"),
                   ("network", f"{cfg.hidden} hidden neurons"),
                   ("games", f"{cfg.games_per_eval} per individual per generation"),
                   ("saving", f"every {plan['save_every']} generations"), ("seed", seed)]
        noun = "generations"
    else:
        summary = [("method", "reinforcement learning (DQN)"), ("run", where),
                   ("rounds", f"{plan['count']} more, each with {cfg.episodes_per_round} training games and {cfg.eval_games} test games"),
                   ("network", f"{cfg.hidden} hidden neurons"),
                   ("learning", f"discount {cfg.gamma}, learning rate {cfg.lr}, batch {cfg.batch_size}, one update per {cfg.train_every} steps"),
                   ("memory", f"{cfg.buffer_size:,} transitions, target network refreshed every {cfg.target_sync:,} steps"),
                   ("exploration", f"random actions fall from 100% to 5% over {cfg.eps_decay_steps:,} steps"),
                   ("saving", f"a checkpoint every {plan['save_every']} rounds, plus best.npz"), ("seed", seed)]
        noun = "rounds"
    warning = _delete_warning(name, info, noun) if plan["reset"] else None
    return plan if _confirm(summary, io, warning) else None


def _evolution_plan(runs_dir, io):
    from ml.neuroevolution.evolve import Config, run_info

    out = io["out"]
    existing = {n: run_info(os.path.join(runs_dir, n)) for n in _list_runs(runs_dir)}
    existing = {n: info for n, info in existing.items() if info}
    if existing:
        out("Existing neuroevolution runs: " + ", ".join(f"{n} (trained {i.generation} generations)" for n, i in existing.items()) + "\n")

    name, info, reset = _pick_run(runs_dir, io, existing, "default", "dqn", "DQN, which is the other kind of training", "generations")
    fixed = info is not None and not reset

    generations = ask("How many generations to train", 100, positive_int, **io)
    if fixed:
        pop, hidden = info.pop_size, info.hidden
        out(f"  (population {pop} and {hidden} hidden neurons are fixed by this run)")
    else:
        pop = ask("Population size: individuals per generation", 50, positive_int, **io)
        hidden = ask("Hidden neurons in the network", 12, positive_int, **io)

    def elite_ok(text):
        n = non_negative_int(text)
        if n > pop:
            raise ValueError(f"can't be more than the population size ({pop})")
        return n

    elite = ask("Elites: the top N are copied unchanged into the next generation", min(5, pop), elite_ok, **io)

    def parents_ok(text):
        n = positive_int(text)
        if n < elite:
            raise ValueError(f"must be at least the number of elites ({elite}), since the elites are parents too")
        if n > pop:
            raise ValueError(f"can't be more than the population size ({pop})")
        return n

    parents = ask("Parents: the top N are allowed to have children", min(max(10, elite), pop), parents_ok, **io)
    games = ask("Games each individual plays per generation", 3, positive_int, **io)
    save_every = ask("Save the whole generation every N generations", 5, positive_int, **io)
    seed = None if fixed else ask("Random seed (same seed = same run)", 0, any_int, **io)
    verbose = ask_yes_no("Print every individual's fitness each generation?", False, **io)

    plan = {"method": "evolution", "run": name, "reset": reset, "count": generations, "seed": seed,
            "save_every": save_every, "verbose": verbose,
            "config": Config(pop_size=pop, elite=elite, n_parents=parents, hidden=hidden, games_per_eval=games)}
    return _confirm_train(plan, info, fixed, io)


def _dqn_plan(runs_dir, io):
    from ml.rl.train import DQNConfig, run_info

    out = io["out"]
    existing = {n: run_info(os.path.join(runs_dir, n)) for n in _list_runs(runs_dir)}
    existing = {n: info for n, info in existing.items() if info}
    if existing:
        out("Existing DQN runs: " + ", ".join(f"{n} (trained {i.round} rounds)" for n, i in existing.items()) + "\n")

    name, info, reset = _pick_run(runs_dir, io, existing, "dqn", "evolution", "neuroevolution, which is the other kind of training", "rounds")
    fixed = info is not None and not reset

    rounds = ask("How many rounds to train", 100, positive_int, **io)
    episodes = ask("Training games per round", 10, positive_int, **io)
    if fixed:
        hidden = info.hidden
        out(f"  ({hidden} hidden neurons and a memory of {info.buffer_size:,} transitions are fixed by this run)")
    else:
        hidden = ask("Hidden neurons in the network", 128, positive_int, **io)
    save_every = ask("Save a checkpoint every N rounds", 5, positive_int, **io)
    seed = None if fixed else ask("Random seed (same seed = same run)", 0, any_int, **io)

    gamma, lr, batch, train_every, target_sync, eps_decay, eval_games = 0.99, 0.001, 64, 4, 4000, 50000, 3
    buffer = info.buffer_size if fixed else 100000
    if ask_yes_no("Change the advanced settings (discount, learning rate, exploration, ...)?", False, **io):
        gamma = ask("Discount gamma: how much a reward one step later counts (0 up to below 1)", gamma, gamma_value, **io)
        lr = ask("Learning rate", lr, positive_float, **io)

        def batch_ok(text):
            n = positive_int(text)
            if n > buffer:
                raise ValueError(f"can't be more than the memory size ({buffer:,})")
            return n

        batch = ask("Batch size: transitions used per training step", batch, batch_ok, **io)
        if not fixed:
            def buffer_ok(text):
                n = positive_int(text)
                if n < batch:
                    raise ValueError(f"must hold at least one batch ({batch:,} transitions)")
                return n

            buffer = ask("Memory: how many recent transitions are kept", buffer, buffer_ok, **io)
        train_every = ask("Game steps per training step", train_every, positive_int, **io)
        target_sync = ask("Game steps between refreshing the frozen target network", target_sync, positive_int, **io)
        eps_decay = ask("Exploration: game steps for random actions to fall from 100% to 5%", eps_decay, positive_int, **io)
        eval_games = ask("Test games per round", eval_games, positive_int, **io)

    warmup = max(batch, min(1000, buffer))      # training starts once the memory holds this much; never more than it can hold
    plan = {"method": "dqn", "run": name, "reset": reset, "count": rounds, "seed": seed, "save_every": save_every,
            "verbose": False,
            "config": DQNConfig(hidden=hidden, gamma=gamma, lr=lr, batch_size=batch, buffer_size=buffer,
                                train_every=train_every, target_sync=target_sync, warmup=warmup,
                                eps_decay_steps=eps_decay, episodes_per_round=episodes, eval_games=eval_games)}
    return _confirm_train(plan, info, fixed, io)


# --- watching ----------------------------------------------------------------

WATCH_TYPE_LABELS = [
    "single: one player (an evolution or DQN player, or the perfect bot)",
    "single VS: an evolution player on the left, a DQN player on the right",
    "multiple: several generations or rounds of one kind of training, side by side",
    "multiple VS: evolution generations on the left, DQN rounds on the right, one pair per row",
]
WATCH_TYPES = ["single", "single_vs", "multiple", "multiple_vs"]
METHOD_NAME = {"evolution": "neuroevolution", "dqn": "DQN"}
UNIT_OF = {"evolution": "gen", "dqn": "round"}


def _show_scores(run_dir, saved, unit, out):
    """List every saved checkpoint with its score, so you can choose by result."""
    from ml.brains import best_checkpoint, saved_scores
    scores = saved_scores(run_dir)
    if unit == "round":
        word, meaning = "eval", "the test score"
    else:
        word, meaning = "best", "its best individual's training fitness"
    items = [f"{g}: {word} {scores[g]:.1f}" for g in saved if g in scores]
    out(f"  Scores ({word} = {meaning}):")
    for i in range(0, len(items), 4):
        out("    " + "   ".join(items[i:i + 4]))
    if unit == "round":
        confirmed = best_checkpoint(run_dir)
        if confirmed:
            out(f"    best: round {confirmed[0]}, eval {confirmed[1]:.1f} (confirmed on fresh games)")
    out("")


def _pick_checkpoint(run_dir, saved, unit, io):
    """Ask which checkpoint of a run to use: a number, or best for a DQN run. Returns a number or 'best'."""
    noun = NOUN[unit]
    has_best = os.path.exists(os.path.join(run_dir, "best.npz"))
    _show_scores(run_dir, saved, unit, io["out"])

    def convert(text):
        if has_best and text.lower() == "best":
            return "best"
        n = any_int(text)
        if n not in saved:
            options = ", ".join(map(str, saved)) + (", best" if has_best else "")
            raise ValueError(f"{noun} {n} was not saved; choose one of: {options}")
        return n

    options = "a number, or best" if has_best else "a number"
    default = "best" if has_best else saved[-1]
    return ask(f"Which {noun} ({options})", default, convert, shown=str(default), **io)


def _runs_of(runs_dir, method):
    """{run name: saved generations} for every run of one kind that has saved any."""
    from ml.brains import saved_generations
    from ml.runs import checkpoint_kind
    found = {}
    for n in _list_runs(runs_dir):
        folder = os.path.join(runs_dir, n)
        gens = saved_generations(folder)
        if gens and checkpoint_kind(folder) == method:
            found[n] = gens
    return found


def _pick_run_of(runs_dir, method, io, title):
    """Ask which run of one kind to use. Returns (name, run folder, saved generations) or None if there are none."""
    runs = _runs_of(runs_dir, method)
    if not runs:
        io["out"](f"  (there are no {METHOD_NAME[method]} runs with saved {NOUN[UNIT_OF[method]]}s to watch yet)")
        return None
    names = sorted(runs, key=lambda n: os.path.getmtime(os.path.join(runs_dir, n)), reverse=True)
    labels = [f"run '{n}': {len(runs[n])} saved {NOUN[UNIT_OF[method]]}s (latest is {runs[n][-1]})" for n in names]
    pick = ask_choice(title, labels, 1, **io)
    name = names[pick - 1]
    return name, os.path.join(runs_dir, name), runs[name]


def _spread(items, n):
    """About n of the items, spread evenly from the first to the last."""
    if n >= len(items):
        return list(items)
    if n <= 1:
        return [items[-1]]
    return list(dict.fromkeys(items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)))


def _parse_generations(text, saved, noun, allow_best, count=None):
    """A list of generation/round numbers typed by the user. 'latest' is the highest saved one;
    'best' (the best checkpoint) is allowed when asked."""
    items = []
    for token in text.replace(",", " ").split():
        if token.lower() == "latest":
            items.append(saved[-1])
        elif token.lower() == "best":
            if not allow_best:
                raise ValueError("there is no best checkpoint for this run (only DQN runs have one)")
            items.append("best")
        else:
            items.append(any_int(token))
    if count is not None and len(items) != count:
        raise ValueError(f"list exactly {count} {noun}s, one per row")
    if count is None and len(items) < 2:
        raise ValueError(f"list at least two {noun}s")
    if len(set(items)) != len(items):
        raise ValueError(f"a {noun} is listed twice")
    missing = [g for g in items if g != "best" and g not in saved]
    if missing:
        raise ValueError(f"not saved: {', '.join(map(str, missing))}. Saved: {', '.join(map(str, saved))}")
    return items


def _grid_shape(n, io):
    """Ask for rows and columns of a grid of n cells (default: one row)."""
    from game_interface.grid import resolve_size

    def size_ok(text):
        parts = text.replace("x", " ").replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("give two numbers: rows then columns, like 2 2")
        return resolve_size(n, (any_int(parts[0]), any_int(parts[1])))

    return ask(f"Grid shape as rows columns (must make {n} cells)", (1, n), size_ok, shown=f"1 {n}", **io)


def _ask_speed(io):
    return ask("Playback speed", 1.0, playback_speed, shown="1", **io)


def _ask_seed(io):
    return ask("Game seed (Enter for a random one)", None, any_int, shown="random", **io)


def _spec_label(spec):
    """'perfect', RUN_DIR:N, RUN_DIR:best, or a checkpoint file, described for a person."""
    if spec == "perfect":
        return "the perfect bot"
    from ml.brains import run_unit
    if spec.endswith(".npz"):
        run_dir, base = os.path.dirname(spec), os.path.basename(spec)
        which = "best checkpoint" if base == "best.npz" else f"{NOUN[run_unit(run_dir)]} {int(base[4:-4])}"
    else:
        run_dir, which = spec.rsplit(":", 1)
        which = "best checkpoint" if which == "best" else f"{NOUN[run_unit(run_dir)]} {which}"
    return f"run '{os.path.basename(os.path.normpath(run_dir))}', {which}"


def _side_label(specs):
    """'run X: generations 0, 50, 99' for the specs of one side of a VS grid."""
    run = os.path.basename(os.path.normpath(specs[0].rsplit(":", 1)[0]))
    picks = [s.rsplit(":", 1)[1] for s in specs]
    from ml.brains import run_unit
    noun = NOUN[run_unit(specs[0].rsplit(":", 1)[0])]
    return f"run '{run}', {noun}s {', '.join(picks)}"


def _watch_summary(plan):
    """The summary shown before the go-ahead, the same whether the plan came from questions or the file."""
    seed = plan["seed"]
    games = "random obstacles" if seed is None else f"obstacles from seed {seed}"
    speed = f"{plan['speed']:g}x"
    kind = plan["type"]
    if kind == "single":
        if plan["brain"] == "perfect":
            return [("watching", "the perfect bot"), ("games", games), ("speed", speed)]
        individual = f", individual #{plan['rank']}" if plan["rank"] != 1 else ""
        games = "the games it was scored on" if plan["replay"] else games
        return [("watching", _spec_label(plan["brain"]) + individual), ("games", games), ("speed", speed)]
    if kind == "single_vs":
        left, right = plan["specs"]
        return [("left", _spec_label(left)), ("right", _spec_label(right)), ("grid", "1 x 2"), ("games", games + ", the same for both"),
                ("speed", speed)]
    if kind == "multiple":
        from ml.brains import run_unit
        run = os.path.basename(os.path.normpath(plan["run_dir"]))
        noun = NOUN[run_unit(plan["run_dir"])]
        individual = f", individual #{plan['rank']}" if plan["rank"] != 1 else ""
        return [("watching", f"run '{run}', {noun}s {', '.join(map(str, plan['generations']))}{individual}"),
                ("grid", f"{plan['rows']} x {plan['cols']}, earliest top-left"),
                ("games", games + ", the same for every cell"), ("speed", speed)]
    left, right = plan["specs"][0::2], plan["specs"][1::2]
    return [("left", _side_label(left)), ("right", _side_label(right)),
            ("grid", f"{plan['rows']} x {plan['cols']}, one row per pair"), ("games", games + ", the same for every cell"),
            ("speed", speed)]


def _ask_single(runs_dir, io, method):
    """The questions for watching one evolution or DQN player (the kind was chosen before)."""
    from ml.brains import load_brain, load_generation
    found = _pick_run_of(runs_dir, method, io, f"Which {METHOD_NAME[method]} run?")
    if found is None:
        return None
    name, run_dir, saved = found
    chosen = _pick_checkpoint(run_dir, saved, UNIT_OF[method], io)
    rank = 1
    if chosen == "best":
        brain = os.path.join(run_dir, "best.npz")
    else:
        brain = os.path.join(run_dir, f"gen_{chosen:04d}.npz")
        individuals = load_generation(run_dir, chosen, 1).n_individuals or 1
        if individuals > 1:
            def rank_ok(text):
                r = positive_int(text)
                load_generation(run_dir, chosen, r)     # raises if that generation has fewer individuals
                return r
            rank = ask(f"Which individual of that {NOUN[UNIT_OF[method]]} (1 = best, 2 = second best, ...)", 1, rank_ok, **io)
    replay = False
    if load_brain(brain).seeds:
        replay = ask_yes_no("Replay the exact games it was scored on during training (instead of new random games)?", False, **io)
    seed = None if replay else _ask_seed(io)
    speed = _ask_speed(io)
    return {"kind": "single", "type": "single", "brain": brain, "rank": rank, "replay": replay, "seed": seed, "speed": speed}


def _ask_single_vs(runs_dir, io):
    sides = []
    for method, title in (("evolution", "Left: which neuroevolution run?"), ("dqn", "Right: which DQN run?")):
        found = _pick_run_of(runs_dir, method, io, title)
        if found is None:
            return None
        name, run_dir, saved = found
        chosen = _pick_checkpoint(run_dir, saved, UNIT_OF[method], io)
        sides.append(f"{run_dir}:{chosen}")
    seed = _ask_seed(io)
    speed = _ask_speed(io)
    return {"kind": "vs", "type": "single_vs", "specs": sides, "rank": 1, "rows": 1, "cols": 2, "seed": seed, "speed": speed}


def _ask_multiple(runs_dir, io, method):
    """The questions for several generations or rounds of one kind of training (the kind was chosen before)."""
    from ml.brains import load_brain, load_generation
    found = _pick_run_of(runs_dir, method, io, f"Which {METHOD_NAME[method]} run?")
    if found is None:
        return None
    name, run_dir, saved = found
    noun = NOUN[UNIT_OF[method]]
    _show_scores(run_dir, saved, UNIT_OF[method], io["out"])
    has_best = method == "dqn" and os.path.exists(os.path.join(run_dir, "best.npz"))
    picks = _spread(saved, 3) + ["best"] if has_best else _spread(saved, 4)
    hint = f", or latest{', or best for the best checkpoint' if has_best else ''}"
    gens = ask(f"{noun.capitalize()}s to compare (space-separated{hint})", picks,
               lambda t: _parse_generations(t, saved, noun, allow_best=has_best), shown=" ".join(map(str, picks)), **io)
    rows, cols = _grid_shape(len(gens), io)
    rank = 1
    if (load_generation(run_dir, saved[0], 1).n_individuals or 1) > 1:
        rank = ask(f"Which individual of each {noun} (1 = best)", 1, lambda t: positive_int(t), **io)
    seed = _ask_seed(io)
    speed = _ask_speed(io)
    return {"kind": "multiple", "type": "multiple", "run_dir": run_dir, "generations": gens, "rank": rank,
            "rows": rows, "cols": cols, "seed": seed, "speed": speed}


def _ask_side_generations(runs_dir, method, io, title, count=None, default_count=3):
    """Pick a run of one kind, then which of its generations/rounds. Returns (run folder, numbers) or None."""
    found = _pick_run_of(runs_dir, method, io, title)
    if found is None:
        return None
    name, run_dir, saved = found
    noun = NOUN[UNIT_OF[method]]
    has_best = os.path.exists(os.path.join(run_dir, "best.npz")) and method == "dqn"
    need = count or default_count
    room = len(saved) + (1 if has_best else 0)
    if need > room:
        io["out"](f"  ! run '{name}' has only {room} saved {noun}s to pick from, but {need} are needed")
        return None
    _show_scores(run_dir, saved, UNIT_OF[method], io["out"])
    if has_best:
        picks = _spread(saved, need - 1) + ["best"]
        hint = ", best = the best checkpoint"
    else:
        picks = _spread(saved, need)
        hint = ""
    shown = " ".join(map(str, picks))
    gens = ask(f"{noun.capitalize()}s to show, {need} of them (space-separated{hint})", picks,
               lambda t: _parse_generations(t, saved, noun, allow_best=has_best, count=count if count else need),
               shown=shown, **io)
    return run_dir, gens


def _ask_multiple_vs(runs_dir, io):
    io["out"]("  Left: neuroevolution generations. Right: DQN rounds. Each row pairs one of each, and both sides need the same count.")
    left = _ask_side_generations(runs_dir, "evolution", io, "Left: which neuroevolution run?", default_count=3)
    if left is None:
        return None
    right = _ask_side_generations(runs_dir, "dqn", io, "Right: which DQN run?", count=len(left[1]))
    if right is None:
        return None
    specs = []
    for e, d in zip(left[1], right[1]):
        specs.append(f"{left[0]}:{e}")
        specs.append(f"{right[0]}:{d}")
    seed = _ask_seed(io)
    speed = _ask_speed(io)
    return {"kind": "vs", "type": "multiple_vs", "specs": specs, "rank": 1, "rows": len(left[1]), "cols": 2,
            "seed": seed, "speed": speed}


def _perfect_plan(io):
    """The perfect bot has nothing to choose except the obstacles and the speed."""
    seed = _ask_seed(io)
    speed = _ask_speed(io)
    return {"kind": "single", "type": "single", "brain": "perfect", "rank": 1, "replay": False, "seed": seed, "speed": speed}


def watch_wizard(runs_dir, *, input_fn=input, out=print, defaults_path=None):
    """Ask the type of watching and, for single and multiple, which kind (evolution or DQN); only then offer
    the saved settings or the questions. The settings file never decides the kind."""
    io = {"input_fn": input_fn, "out": out}
    out("Watch players. Press Enter to accept the value in [brackets].\n")
    kind = WATCH_TYPES[ask_choice("What do you want to watch?", WATCH_TYPE_LABELS, 1, **io) - 1]

    mode = None
    if kind == "single":
        mode = ["evolution", "dqn", "perfect"][ask_choice("Which kind of player?", [
            "evolution: a generation of a neuroevolution run",
            "DQN: a round of a reinforcement-learning run",
            "the perfect bot (a hand-written rule, no learning)",
        ], 1, **io) - 1]
    elif kind == "multiple":
        mode = ["evolution", "dqn"][ask_choice("Which kind of training?", [
            "evolution: generations of a neuroevolution run",
            "DQN: rounds of a reinforcement-learning run",
        ], 1, **io) - 1]
    out("")

    if mode == "perfect":
        plan = _perfect_plan(io)
        return plan if _confirm(_watch_summary(plan), io) else None

    label = {"single": "single", "single_vs": "single VS", "multiple": "multiple", "multiple_vs": "multiple VS"}[kind]
    if mode:
        label += f" {'evolution' if mode == 'evolution' else 'DQN'}"
    if _use_settings_file(defaults_path, io, f"Use the saved {label} settings", default=2):
        try:
            plan = watch_plan_from_file(defaults_path, runs_dir, kind, mode)
        except SettingsError as e:
            out(f"\n  ! {e}\n  Fix the file, or answer the questions now.")
        else:
            return plan if _confirm(_watch_summary(plan), io) else None
    out("")
    if kind in ("single", "multiple"):
        plan = (_ask_single if kind == "single" else _ask_multiple)(runs_dir, io, mode)
    else:
        plan = (_ask_single_vs if kind == "single_vs" else _ask_multiple_vs)(runs_dir, io)
    if plan is None:
        return None
    return plan if _confirm(_watch_summary(plan), io) else None
