"""Carry out a training plan: the dictionary that ml/run/_interactive.py's train wizard returns.

  {"method": "evolution" | "dqn", "run": folder name, "reset": bool, "count": generations or rounds,
   "seed": int | None, "save_every": int, "verbose": bool, "config": Config | DQNConfig}
"""
import os
import shutil


def run_training(plan: dict, runs_dir: str, log=print):
    run = plan["run"]
    if os.sep in run or run in ("", ".", ".."):
        raise ValueError("the run name must be a plain folder name")
    run_dir = os.path.join(runs_dir, run)
    if plan["reset"]:
        shutil.rmtree(run_dir, ignore_errors=True)

    if plan["method"] == "evolution":
        from ml.neuroevolution.evolve import train
        train(plan["config"], plan["count"], run_dir, seed=plan["seed"], save_every=plan["save_every"],
              verbose=plan["verbose"], log=log)
    elif plan["method"] == "dqn":
        from ml.rl.train import train
        train(plan["config"], plan["count"], run_dir, seed=plan["seed"], save_every=plan["save_every"], log=log)
    else:
        raise ValueError(f"unknown kind of training: {plan['method']!r}")
