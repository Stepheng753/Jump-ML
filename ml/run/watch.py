#!/usr/bin/env python3
"""Watch players play in the real game window. Run it and answer the questions (press Enter to
accept the default in [brackets]).

You can watch:
  - one saved generation or round of a trained run (any individual of it, or a DQN run's best.npz),
  - several generations or rounds of a run side by side (earliest top-left), all on the same obstacles,
  - VS mode: players picked from different runs (evolution and DQN together) and the perfect bot,
    all on the same obstacles,
  - the perfect bot on its own.

Wherever it asks which generation or round, it lists them all with their scores first
(best = a generation's best individual's training fitness; eval = a DQN round's test score).
By default every game uses fresh random obstacles; you can choose a seed, or replay the games a
generation was scored on.

Keys in the window: P pause, N step one frame (while paused), [ / ] slower / faster,
SPACE skip to the next game (next round in a grid), Esc quit.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))   # _venv.py, the launcher helper every script shares, lives there
import _venv  # noqa: E402
from _interactive import watch_wizard  # noqa: E402

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

_venv.ensure(__file__)
from game_interface.plan import build_watch, launch  # noqa: E402

try:
    plan = watch_wizard(os.path.join(_venv.ROOT, "runs"), defaults_path=os.path.join(HERE, "default.json"))
except (EOFError, KeyboardInterrupt):
    sys.exit("\nCancelled.")
if plan is None:
    sys.exit("Cancelled.")

try:
    job = build_watch(plan)
except (ValueError, FileNotFoundError) as e:
    sys.exit(f"error: {e}")
launch(job)
