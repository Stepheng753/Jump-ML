"""Which training method made a run folder?"""
import glob
import os
from typing import Optional

import numpy as np


def checkpoint_kind(run_dir: str) -> Optional[str]:
    """'dqn' or 'evolution' from a run's saved checkpoints (what the watcher needs), or None if it has none."""
    files = sorted(glob.glob(os.path.join(run_dir, "gen_*.npz")))
    if not files:
        return None
    data = np.load(files[-1])
    return "dqn" if "kind" in data.files and str(data["kind"]) == "dqn" else "evolution"


def run_method(run_dir: str) -> Optional[str]:
    """'evolution', 'dqn', or None if the folder has no saved training state."""
    path = os.path.join(run_dir, "state.npz")
    if not os.path.exists(path):
        return None
    files = np.load(path).files
    if "theta" in files:
        return "dqn"
    return "evolution" if "population" in files else None
