"""Read default.json: saved answers to the questions of train.py (its "train" part) and watch.py (its "watch" part).

Each loader checks every value with the same rules the questions use, reports ALL the problems at once
(naming the setting), and returns the same kind of plan the wizards return:

  train_plan_from_file(path, runs_dir) -> (plan, info, fixed)   plan as in _interactive.train_wizard;
        info describes an existing run of this kind (or None); fixed means "continuing it"
  watch_plan_from_file(path, runs_dir) -> plan                  plan as in _interactive.watch_wizard

Keys starting with an underscore (like "_note") are ignored, so the files can carry notes.
"""
import argparse
import json
import os

from _args import any_int, gamma_value, non_negative_int, playback_speed, positive_float, positive_int, run_name

TRAIN_KINDS = {"evolution": "neuroevolution", "dqn": "DQN"}


NOT_USED_KIND = "not used: you choose neuroevolution or DQN when the script runs, never in this file"


class SettingsError(ValueError):
    """The settings file is missing, is not valid JSON, or has values that don't make sense."""


class _Reader:
    def __init__(self, path, part):
        """Reads the `part` ("train" or "watch") of the settings file; messages name settings as 'part.setting'."""
        self.filename = os.path.basename(path)
        self.errors = []
        self.prefix = f"{part}."
        try:
            with open(path) as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise SettingsError(f"{self.filename} was not found in {os.path.dirname(os.path.abspath(path))}")
        except json.JSONDecodeError as e:
            raise SettingsError(f"{self.filename} is not valid JSON: {e}")
        if not isinstance(self.data, dict):
            raise SettingsError(f"{self.filename} must contain a JSON object ({{ ... }})")
        for unknown in sorted(k for k in self.data if k not in ("train", "watch") and not k.startswith("_")):
            self.errors.append(f"{unknown}: unknown section (only \"train\" and \"watch\" are used)")
        if not isinstance(self.data.get(part), dict):
            raise SettingsError(f"{self.filename} has no \"{part}\" section")
        self.data = self.data[part]

    def fail(self, where, problem):
        self.errors.append(f"{self.prefix}{where}: {problem}")

    def section(self, data, key, known, hints=None):
        """data[key] as a dictionary, with a complaint about any setting it doesn't know.
        hints maps a setting name to what to say about it (for settings that moved or are no longer used)."""
        section = data.get(key)
        if not isinstance(section, dict):
            self.fail(key, "this section is missing")
            return {}
        for unknown in sorted(k for k in section if k not in known and not k.startswith("_")):
            self.fail(f"{key}.{unknown}", (hints or {}).get(unknown, "unknown setting (a typo?)"))
        return section

    def number(self, section, key, check, where, default=...):
        value = section.get(key, default)
        if value is ...:
            self.fail(where, "this setting is missing")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(where, f"must be a number (got {json.dumps(value)})")
            return None
        try:
            return check(str(value))
        except argparse.ArgumentTypeError as e:
            self.fail(where, str(e))
            return None

    def optional_seed(self, section, key, where):
        return None if section.get(key) is None else self.number(section, key, any_int, where)

    def text(self, section, key, check, where):
        value = section.get(key)
        if not isinstance(value, str):
            self.fail(where, f"must be text (got {json.dumps(value)})")
            return None
        try:
            return check(value)
        except argparse.ArgumentTypeError as e:
            self.fail(where, str(e))
            return None

    def flag(self, section, key, where, default=False):
        value = section.get(key, default)
        if not isinstance(value, bool):
            self.fail(where, f"must be true or false (got {json.dumps(value)})")
            return default
        return value

    def choice(self, section, key, options, where):
        value = section.get(key)
        if value not in options:
            self.fail(where, f"must be one of {', '.join(json.dumps(o) for o in options)} (got {json.dumps(value)})")
            return None
        return value

    def raise_if_any(self):
        if self.errors:
            raise SettingsError(f"{self.filename} has problems:\n  - " + "\n  - ".join(self.errors))


# --- the train part of default.json ----------------------------------------------------------

EVOLUTION_KEYS = {"run", "generations", "population", "elites", "parents", "hidden", "games_per_individual",
                  "save_every", "seed", "verbose", "existing_run"}
DQN_KEYS = {"run", "rounds", "training_games_per_round", "hidden", "save_every", "seed", "existing_run", "gamma",
            "learning_rate", "batch_size", "memory_size", "train_every", "target_sync", "exploration_steps",
            "test_games_per_round"}


def train_plan_from_file(path, runs_dir, method):
    """The train part of the file, for the kind of training `method` ('evolution' or 'dqn').

    The kind is always chosen by the question asked before this one; the file never decides it.
    """
    from ml.neuroevolution.evolve import Config
    from ml.neuroevolution.evolve import run_info as evolution_info
    from ml.rl.train import DQNConfig
    from ml.rl.train import run_info as dqn_info
    from ml.runs import run_method

    r = _Reader(path, "train")
    if method not in TRAIN_KINDS:
        raise SettingsError(f"unknown kind of training: {method!r}")
    for unknown in sorted(k for k in r.data if k not in ("evolution", "dqn") and not k.startswith("_")):
        r.fail(unknown, NOT_USED_KIND if unknown == "method" else "unknown setting (a typo?)")
    r.raise_if_any()
    s = r.section(r.data, method, EVOLUTION_KEYS if method == "evolution" else DQN_KEYS)
    where = lambda key: f"{method}.{key}"  # noqa: E731

    name = r.text(s, "run", run_name, where("run"))
    save_every = r.number(s, "save_every", positive_int, where("save_every"))
    seed = r.optional_seed(s, "seed", where("seed"))
    existing_run = r.choice(s, "existing_run", ["continue", "start_over"], where("existing_run"))
    hidden = r.number(s, "hidden", positive_int, where("hidden"))

    if method == "evolution":
        count = r.number(s, "generations", positive_int, where("generations"))
        pop = r.number(s, "population", positive_int, where("population"))
        elite = r.number(s, "elites", non_negative_int, where("elites"))
        parents = r.number(s, "parents", positive_int, where("parents"))
        games = r.number(s, "games_per_individual", positive_int, where("games_per_individual"))
        verbose = r.flag(s, "verbose", where("verbose"))
    else:
        count = r.number(s, "rounds", positive_int, where("rounds"))
        episodes = r.number(s, "training_games_per_round", positive_int, where("training_games_per_round"))
        gamma = r.number(s, "gamma", gamma_value, where("gamma"))
        lr = r.number(s, "learning_rate", positive_float, where("learning_rate"))
        batch = r.number(s, "batch_size", positive_int, where("batch_size"))
        memory = r.number(s, "memory_size", positive_int, where("memory_size"))
        train_every = r.number(s, "train_every", positive_int, where("train_every"))
        target_sync = r.number(s, "target_sync", positive_int, where("target_sync"))
        eps_decay = r.number(s, "exploration_steps", positive_int, where("exploration_steps"))
        test_games = r.number(s, "test_games_per_round", positive_int, where("test_games_per_round"))
    r.raise_if_any()

    # the other kind of training owns this run name?
    run_dir = os.path.join(runs_dir, name)
    other = "dqn" if method == "evolution" else "evolution"
    if run_method(run_dir) == other:
        r.fail(where("run"), f"'{name}' was trained with {TRAIN_KINDS[other]}, the other kind of training; pick another run name")
        r.raise_if_any()
    info = (evolution_info if method == "evolution" else dqn_info)(run_dir)
    reset = existing_run == "start_over" and os.path.isdir(run_dir)
    fixed = info is not None and not reset

    # continuing: the settings that can't change must match what the run was started with
    if fixed:
        if info.hidden != hidden:
            r.fail(where("hidden"), f"run '{name}' was started with {info.hidden} hidden neurons; change this to {info.hidden}, "
                                    f"or set existing_run to \"start_over\"")
        if method == "evolution" and info.pop_size != pop:
            r.fail(where("population"), f"run '{name}' was started with a population of {info.pop_size}; change this to "
                                        f"{info.pop_size}, or set existing_run to \"start_over\"")
        if method == "dqn" and info.buffer_size != memory:
            r.fail(where("memory_size"), f"run '{name}' was started with a memory of {info.buffer_size:,} transitions; change "
                                         f"this to {info.buffer_size}, or set existing_run to \"start_over\"")
        r.raise_if_any()

    try:
        if method == "evolution":
            config = Config(pop_size=pop, elite=elite, n_parents=parents, hidden=hidden, games_per_eval=games)
        else:
            warmup = max(batch, min(1000, memory))
            config = DQNConfig(hidden=hidden, gamma=gamma, lr=lr, batch_size=batch, buffer_size=memory,
                               train_every=train_every, target_sync=target_sync, warmup=warmup, eps_decay_steps=eps_decay,
                               episodes_per_round=episodes, eval_games=test_games)
    except ValueError as e:
        r.fail(method, str(e))
        r.raise_if_any()

    plan = {"method": method, "run": name, "reset": reset, "count": count, "seed": None if fixed else (0 if seed is None else seed),
            "save_every": save_every, "verbose": verbose if method == "evolution" else False, "config": config}
    return plan, info, fixed


# --- the watch part of default.json ----------------------------------------------------------

WATCH_TYPES = ("single", "single_vs", "multiple", "multiple_vs")
TYPE_KEYS = {
    "single": {"evolution", "dqn", "individual", "replay"},
    "single_vs": {"evolution", "dqn"},
    "multiple": {"evolution", "dqn", "individual", "rows", "cols"},
    "multiple_vs": {"evolution", "dqn", "rows", "cols"},
}
MOVED = "moved: put it inside the 'evolution' and 'dqn' sub-sections; you choose which kind when the script runs"
MOVED_HINTS = {"single": {"training": NOT_USED_KIND, "run": MOVED, "checkpoint": MOVED},
               "multiple": {"training": NOT_USED_KIND, "run": MOVED, "generations": MOVED}}
SIDE_KEYS = {"run", "checkpoint"}
MULTIPLE_VS_SIDE_KEYS = {"run", "generations"}


def watch_plan_from_file(path, runs_dir, kind, mode=None):
    """The watch part of the file, for one of the four types of watching. Returns a plan (see
    game_interface/plan.py), or raises SettingsError listing every problem.

    single and multiple also need `mode`, the kind of player the user chose just before: 'evolution' or 'dqn'
    ('perfect' too, for single). The file never decides it.
    """
    from ml.brains import saved_generations
    from ml.runs import checkpoint_kind

    if kind not in WATCH_TYPES:
        raise SettingsError(f"unknown type of watching: {kind!r}")
    modes = {"single": ("evolution", "dqn", "perfect"), "multiple": ("evolution", "dqn")}.get(kind)
    if modes and mode not in modes:
        raise SettingsError(f"{kind} needs to know the kind to watch: one of {', '.join(modes)} (got {mode!r})")
    r = _Reader(path, "watch")
    for unknown in sorted(k for k in r.data if k not in ("speed", "seed") + WATCH_TYPES and not k.startswith("_")):
        r.fail(unknown, "unknown setting (a typo?)")
    speed = r.number(r.data, "speed", playback_speed, "speed")
    seed = r.optional_seed(r.data, "seed", "seed")
    s = r.section(r.data, kind, TYPE_KEYS[kind], MOVED_HINTS.get(kind))

    def run_of(name, method, where):
        """(run folder, saved generations) of a run of the given method, or None with a problem recorded."""
        if not isinstance(name, str):
            r.fail(where, f"must be the name of a run under runs/ (got {json.dumps(name)})")
            return None
        try:
            run_name(name)
        except argparse.ArgumentTypeError as e:
            r.fail(where, str(e))
            return None
        run_dir = os.path.join(runs_dir, name)
        saved = saved_generations(run_dir)
        if not saved:
            r.fail(where, f"run '{name}' has no saved generations or rounds")
            return None
        actual = checkpoint_kind(run_dir)
        if actual != method:
            r.fail(where, f"run '{name}' is a {TRAIN_KINDS.get(actual, 'unknown')} run, but this needs a {TRAIN_KINDS[method]} run")
            return None
        return run_dir, saved

    def checkpoint(run_dir, saved, value, where):
        """'latest' | 'best' | a saved number, resolved to 'best' or a number; None with a problem recorded."""
        if value == "latest":
            return saved[-1]
        if value == "best":
            if not os.path.exists(os.path.join(run_dir, "best.npz")):
                r.fail(where, f"run '{os.path.basename(run_dir)}' has no best checkpoint (only DQN runs do); use \"latest\" or a number")
                return None
            return "best"
        if isinstance(value, int) and not isinstance(value, bool) and value in saved:
            return value
        r.fail(where, f"must be \"latest\", \"best\" or a saved number (saved: {', '.join(map(str, saved))}; got {json.dumps(value)})")
        return None

    def generations(run_dir, saved, value, where, allow_best, count=None):
        """A list of saved numbers, "latest" (becomes the highest saved number) and, when allowed, "best".
        None with a problem recorded."""
        run = os.path.basename(run_dir)
        has_best = os.path.exists(os.path.join(run_dir, "best.npz"))
        if not isinstance(value, list) or not value:
            r.fail(where, 'must be a list, like [0, 50, "latest"]')
            return None
        items = []
        for v in value:
            if v == "latest":
                items.append(saved[-1])
            elif v == "best" and not allow_best:
                r.fail(where, '"best" is only for DQN runs; use "latest" or a saved number for a neuroevolution run')
                return None
            elif v == "best" and not has_best:
                r.fail(where, f"run '{run}' has no best checkpoint yet")
                return None
            elif v == "best":
                items.append("best")
            elif isinstance(v, int) and not isinstance(v, bool) and v in saved:
                items.append(v)
            else:
                r.fail(where, f"{json.dumps(v)} was not saved in run '{run}'. Saved: {', '.join(map(str, saved))}, latest"
                              + (", best" if allow_best and has_best else ""))
                return None
        if len(set(items)) != len(items):
            r.fail(where, f"a generation is listed twice (\"latest\" is generation {saved[-1]})" if "latest" in value
                   else "a generation is listed twice")
            return None
        if count is not None and len(items) != count:
            r.fail(where, f"must list {count} (one per row), got {len(items)}")
            return None
        if count is None and len(items) < 2:
            r.fail(where, "list at least two generations")
            return None
        return items

    def grid(section, n, where):
        rows, cols = section.get("rows"), section.get("cols")
        if rows is None and cols is None:
            return 1, n
        rows = r.number(section, "rows", positive_int, f"{where}.rows")
        cols = r.number(section, "cols", positive_int, f"{where}.cols")
        if rows and cols and rows * cols != n:
            r.fail(where, f"a {rows} x {cols} grid makes {rows * cols} cells but there are {n} to show: they must match "
                          "(or set rows and cols to null for one row)")
            return None
        return rows, cols

    plan = None
    if kind == "single":
        individual = r.number(s, "individual", positive_int, "single.individual", default=1)
        replay = r.flag(s, "replay", "single.replay")
        brain = None
        if mode == "perfect":
            brain = "perfect"
        else:
            side = r.section(s, mode, SIDE_KEYS)
            found = run_of(side.get("run"), mode, f"single.{mode}.run")
            which = checkpoint(*found, side.get("checkpoint"), f"single.{mode}.checkpoint") if found else None
            if found and which is not None:
                brain = os.path.join(found[0], "best.npz" if which == "best" else f"gen_{which:04d}.npz")
        if replay and seed is not None:
            r.fail("single.replay", "replay uses the games the individual was scored on, so seed must be null")
        r.raise_if_any()
        plan = {"kind": "single", "type": "single", "brain": brain, "rank": individual, "replay": replay,
                "seed": seed, "speed": speed}

    elif kind == "single_vs":
        specs = []
        for side, method in (("evolution", "evolution"), ("dqn", "dqn")):
            sec = r.section(s, side, SIDE_KEYS)
            found = run_of(sec.get("run"), method, f"single_vs.{side}.run")
            which = checkpoint(*found, sec.get("checkpoint"), f"single_vs.{side}.checkpoint") if found else None
            if found and which is not None:
                specs.append(f"{found[0]}:{which}")
        r.raise_if_any()
        plan = {"kind": "vs", "type": "single_vs", "specs": specs, "rank": 1, "rows": 1, "cols": 2,
                "seed": seed, "speed": speed}

    elif kind == "multiple":
        side = r.section(s, mode, MULTIPLE_VS_SIDE_KEYS)
        found = run_of(side.get("run"), mode, f"multiple.{mode}.run")
        gens = generations(*found, side.get("generations"), f"multiple.{mode}.generations", allow_best=(mode == "dqn")) if found else None
        individual = r.number(s, "individual", positive_int, "multiple.individual", default=1)
        shape = grid(s, len(gens), "multiple") if gens else None
        r.raise_if_any()
        if found and gens and shape:
            plan = {"kind": "multiple", "type": "multiple", "run_dir": found[0], "generations": gens,
                    "rank": individual, "rows": shape[0], "cols": shape[1], "seed": seed, "speed": speed}

    else:   # multiple_vs
        sides = {}
        for side, method in (("evolution", "evolution"), ("dqn", "dqn")):
            sec = r.section(s, side, MULTIPLE_VS_SIDE_KEYS)
            found = run_of(sec.get("run"), method, f"multiple_vs.{side}.run")
            count = len(sides["evolution"][1]) if side == "dqn" and "evolution" in sides else None
            gens = generations(*found, sec.get("generations"), f"multiple_vs.{side}.generations",
                               allow_best=(side == "dqn"), count=count) if found else None
            if found and gens:
                sides[side] = (found[0], gens)
        if "evolution" in sides and "dqn" in sides:
            n = len(sides["evolution"][1])
            rows, cols = s.get("rows", n), s.get("cols", 2)
            if rows != n:
                r.fail("multiple_vs.rows", f"must be {n} (one row per pair of generations), or null")
            if cols != 2:
                r.fail("multiple_vs.cols", "must be 2 (left and right), or null")
        r.raise_if_any()
        if "evolution" in sides and "dqn" in sides:
            (evo_dir, evo), (dqn_dir, dqn) = sides["evolution"], sides["dqn"]
            specs = []
            for e, d in zip(evo, dqn):   # the grid fills row by row, so each row is (evolution, DQN)
                specs.append(f"{evo_dir}:{e}")
                specs.append(f"{dqn_dir}:{d}")
            plan = {"kind": "vs", "type": "multiple_vs", "specs": specs, "rank": 1, "rows": len(evo), "cols": 2,
                    "seed": seed, "speed": speed}

    if plan is None:
        raise SettingsError(f"{os.path.basename(path)} has problems:\n  - " + "\n  - ".join(r.errors))
    return plan
