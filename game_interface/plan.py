"""Carry out a watch plan: the dictionary that ml/run/_interactive.py's watch wizard returns.

  {"kind": "single",   "brain": spec, "rank": n, "replay": bool, "seed": n | None, "speed": x}
  {"kind": "multiple", "run_dir": folder, "generations": [...], "rank": n, "rows": r, "cols": c, "seed": ..., "speed": ...}
  {"kind": "vs",       "specs": [...], "rank": n, "rows": r, "cols": c, "seed": ..., "speed": ...}

build_watch loads the players (no window, no pygame); launch opens the window.
"""
from ml.brains import load_brain, load_generation, load_spec

from .grid import by_generation, resolve_size


def build_watch(plan: dict) -> dict:
    """Load the players for a plan. Raises ValueError / FileNotFoundError with a readable message."""
    kind = plan["kind"]
    if kind == "single":
        brain = load_brain(plan["brain"], rank=plan["rank"])
        if plan["replay"] and not getattr(brain, "seeds", None):
            raise ValueError("replay needs a saved generation that recorded its training games; this one has none")
        return {"kind": "single", "brain": brain, "seed": plan["seed"], "speed": plan["speed"], "replay": plan["replay"]}

    if kind == "multiple":
        generations = plan["generations"]
        if len(set(generations)) != len(generations):
            raise ValueError("a generation is listed more than once")
        rows, cols = resolve_size(len(generations), (plan["rows"], plan["cols"]))
        # numbers are shown earliest first; "best" is a checkpoint, not a point in time, so it always comes last
        brains = by_generation([load_generation(plan["run_dir"], g, plan["rank"]) for g in generations if g != "best"])
        if "best" in generations:
            brains.append(load_spec(f"{plan['run_dir']}:best"))
        caption = None
    elif kind == "vs":
        if len(plan["specs"]) < 2:
            raise ValueError("VS mode needs at least two players")
        rows, cols = resolve_size(len(plan["specs"]), (plan["rows"], plan["cols"]))
        brains = [load_spec(s, plan.get("rank", 1)) for s in plan["specs"]]
        caption = "Jump-ML: VS"
    else:
        raise ValueError(f"unknown kind of watching: {kind!r}")
    return {"kind": "grid", "brains": brains, "rows": rows, "cols": cols, "seed": plan["seed"],
            "speed": plan["speed"], "caption": caption}


def launch(job: dict):
    """Open the window for a job made by build_watch."""
    from .watcher import watch, watch_grid

    if job["kind"] == "single":
        watch(job["brain"], seed=job["seed"], speed=job["speed"], replay=job["replay"])
    else:
        watch_grid(job["brains"], job["rows"], job["cols"], seed=job["seed"], speed=job["speed"], caption=job["caption"])
