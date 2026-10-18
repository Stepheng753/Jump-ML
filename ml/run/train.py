#!/usr/bin/env python3
"""Train a player. Run it and answer the questions (press Enter to accept the default in [brackets]).

The first question picks the kind of training:

  1) Neuroevolution: a population of networks plays, the best are kept and bred.
  2) Reinforcement learning (DQN): one network learns while it plays.

Then it asks for that method's settings (run name, how many generations or rounds, ...), shows a
summary, and starts when you confirm. Running it again on an existing run carries on from where it
stopped, or deletes it and starts over (the default). Results go to runs/<run name>/.

Watch the result with ml/run/watch.py. See docs/03 (neuroevolution), docs/05 (DQN) and docs/06 (the two compared).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))   # _venv.py, the launcher helper every script shares, lives there
import _venv  # noqa: E402
from _interactive import train_wizard  # noqa: E402

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

_venv.ensure(__file__)
from ml.plans import run_training  # noqa: E402

runs_dir = os.path.join(_venv.ROOT, "runs")
try:
    plan = train_wizard(runs_dir, defaults_path=os.path.join(HERE, "default.json"))
except (EOFError, KeyboardInterrupt):
    sys.exit("\nCancelled.")
if plan is None:
    sys.exit("Cancelled.")

try:
    run_training(plan, runs_dir)
except ValueError as e:
    sys.exit(f"error: {e}")
