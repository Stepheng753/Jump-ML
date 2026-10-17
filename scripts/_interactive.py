"""Question-and-answer mode for train.py and watch.py (`-i` / `--interactive`).

Each wizard asks for the settings one at a time (press Enter to accept the
default in [brackets]; a bad answer is explained and asked again) and returns
the list of command-line flags it is equivalent to, or None if you decline at
the end. The script then feeds that list to its normal argument parser, so
interactive and command-line modes validate in exactly the same way.
"""
import argparse
import os
import shlex

from _args import any_int, non_negative_int, playback_speed, positive_int


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


def _finish(script, argv, io, warning=None):
    io["out"]("\nEquivalent command:")
    io["out"]("  " + shlex.join([script, *argv]))
    if warning:
        io["out"](f"  ! {warning}")
    return argv if ask_yes_no("Start?", True, **io) else None


def _list_runs(runs_dir):
    if not os.path.isdir(runs_dir):
        return []
    return sorted(n for n in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, n)))


def _rel(path):
    return os.path.relpath(path)   # relative to where you are, so the printed command works as shown


# --- train.py ----------------------------------------------------------------

def train_wizard(runs_dir, *, input_fn=input, out=print):
    from ml.neuroevolution.evolve import run_info

    io = {"input_fn": input_fn, "out": out}
    out("Train a population by neuroevolution. Press Enter to accept the value in [brackets].\n")

    runs = {n: run_info(os.path.join(runs_dir, n)) for n in _list_runs(runs_dir)}
    runs = {n: info for n, info in runs.items() if info}
    if runs:
        out("Existing runs: " + ", ".join(f"{n} (trained {i.generation} generations)" for n, i in runs.items()) + "\n")

    def run_name(text):
        if os.sep in text or text in (".", ".."):
            raise ValueError("use a plain folder name, like xyz")
        return text

    while True:
        name = ask("Run name (saved under runs/)", "default", run_name, **io)
        info, reset = runs.get(name), False
        if info:
            out(f"  '{name}' has already trained {info.generation} generations; training carries on from there.")
            reset = ask_yes_no("  Delete it and start over instead?", True, **io)
            if not reset and info.hidden is None:
                out("  ! This run's network layout can't be read, so it can't be continued. "
                    "Pick another name, or answer y to start over.")
                continue
        break
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

    argv = ["--run", name, "--generations", str(generations), "--pop", str(pop), "--hidden", str(hidden),
            "--elite", str(elite), "--parents", str(parents), "--games", str(games), "--save-every", str(save_every)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if verbose:
        argv.append("--verbose")
    warning = None
    if reset:
        argv.append("--reset")
        warning = (f"This will DELETE runs/{name} ({info.generation} generations trained) "
                   f"before training. Answer n below to cancel.")
    return _finish("scripts/train.py", argv, io, warning)


# --- watch.py ----------------------------------------------------------------

def watch_wizard(runs_dir, *, input_fn=input, out=print):
    from game_interface.grid import resolve_size
    from ml.brains import load_generation, saved_generations

    io = {"input_fn": input_fn, "out": out}
    out("Watch a player. Press Enter to accept the value in [brackets].\n")

    runs = {}
    for n in _list_runs(runs_dir):
        gens = saved_generations(os.path.join(runs_dir, n))
        if gens:
            runs[n] = gens
    names = list(runs)

    labels = [f"run '{n}': {len(g)} saved generations (latest is {g[-1]})" for n, g in runs.items()]
    labels.append("the perfect bot (a hand-written rule, no learning)")
    if not names:
        out("  (no trained runs with saved generations found, so only the perfect bot is available)")
    newest = max(range(len(names)), key=lambda i: os.path.getmtime(os.path.join(runs_dir, names[i]))) + 1 if names else 1
    pick = ask_choice("What do you want to watch?", labels, newest, **io)

    def seed_flag(argv):
        seed = ask("Game seed (Enter for a random one)", None, any_int, shown="random", **io)
        if seed is not None:
            argv += ["--seed", str(seed)]

    if pick == len(labels):
        argv = ["--brain", "perfect"]
        seed_flag(argv)
        argv += ["--speed", str(ask("Playback speed", 1.0, playback_speed, shown="1", **io))]
        return _finish("scripts/watch.py", argv, io)

    name = names[pick - 1]
    run_dir, saved = os.path.join(runs_dir, name), runs[name]
    out(f"\n  Saved generations in '{name}': {', '.join(map(str, saved))}\n")

    mode = 1
    if len(saved) >= 2:
        mode = ask_choice("One generation, or several side by side?",
                          ["one generation", "compare several generations (they all play the same obstacles)"], 1, **io)
    else:
        out("  (only one generation is saved, so there is nothing to compare)")

    if mode == 1:
        def gen_ok(text):
            if text.lower() == "latest":
                return None
            n = any_int(text)
            if n not in saved:
                raise ValueError(f"generation {n} was not saved; choose one of: {', '.join(map(str, saved))}")
            return n

        gen = ask("Which generation (a number, or latest)", None, gen_ok, shown="latest", **io)
        chosen = saved[-1] if gen is None else gen

        def rank_ok(text):
            r = positive_int(text)
            load_generation(run_dir, chosen, r)   # raises ValueError if the generation has fewer individuals
            return r

        rank = ask("Which individual of that generation (1 = best, 2 = second best, ...)", 1, rank_ok, **io)
        argv = ["--brain", _rel(run_dir if gen is None else os.path.join(run_dir, f"gen_{gen:04d}.npz"))]
        if rank != 1:
            argv += ["--rank", str(rank)]
        if load_generation(run_dir, chosen, rank).seeds and ask_yes_no(
                "Replay the exact games it was scored on during training (instead of new random games)?", False, **io):
            argv.append("--replay")
        else:
            seed_flag(argv)
    else:
        picks = [saved[round(i * (len(saved) - 1) / 3)] for i in range(4)] if len(saved) > 4 else saved
        picks = list(dict.fromkeys(picks))

        def gens_ok(text):
            gens = [any_int(t) for t in text.replace(",", " ").split()]
            if len(gens) < 2:
                raise ValueError("list at least two generations")
            if len(set(gens)) != len(gens):
                raise ValueError("a generation is listed twice")
            missing = [g for g in gens if g not in saved]
            if missing:
                raise ValueError(f"not saved: {', '.join(map(str, missing))}. Saved: {', '.join(map(str, saved))}")
            return gens

        gens = ask("Generations to compare (space-separated)", picks, gens_ok, shown=" ".join(map(str, picks)), **io)

        def size_ok(text):
            parts = text.replace("x", " ").replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError("give two numbers: rows then columns, like 2 2")
            return resolve_size(len(gens), (any_int(parts[0]), any_int(parts[1])))

        rows, cols = ask(f"Grid shape as rows columns (must make {len(gens)} cells)", (1, len(gens)), size_ok,
                         shown=f"1 {len(gens)}", **io)

        def rank_ok(text):
            r = positive_int(text)
            for g in gens:
                load_generation(run_dir, g, r)
            return r

        rank = ask("Which individual of each generation (1 = best)", 1, rank_ok, **io)
        argv = ["--brain", _rel(run_dir), "--multiple", *map(str, gens), "--size", str(rows), str(cols)]
        if rank != 1:
            argv += ["--rank", str(rank)]
        seed_flag(argv)

    argv += ["--speed", str(ask("Playback speed", 1.0, playback_speed, shown="1", **io))]
    return _finish("scripts/watch.py", argv, io)
