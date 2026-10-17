#!/usr/bin/env python3
"""Train a population of individuals by neuroevolution (headless, fast):

    scripts/train.py --generations 1 --verbose     # one generation, show every individual's fitness
    scripts/train.py --generations 10              # 10 MORE generations (resumes where it stopped)
    scripts/train.py --generations 50 --reset      # wipe the run and start over
    scripts/train.py --generations 20 --elite 2 --parents 8   # fewer elites, wider parent pool
    scripts/train.py --generations 200 --pop 80 --hidden 24 --run big
    scripts/train.py -i                            # interactive: asks for each setting (default 100 generations)

--generations N means "run N more generations". The population is saved to
runs/<run>/state.npz after every call. The whole generation (all individuals, ranked) is saved every 5 generations
(and after the last one trained) to runs/<run>/gen_XXXX.npz; watch them with
scripts/watch.py.
"""
import argparse
import os
import shutil
import sys

import _venv
from _args import non_negative_int, positive_int
from _interactive import train_wizard

_venv.ensure(__file__)
from ml.neuroevolution.evolve import Config, train  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--generations", type=positive_int, default=10, help="how many MORE generations to run")
p.add_argument("--pop", type=positive_int, default=50, help="individuals per generation")
p.add_argument("--elite", type=non_negative_int, default=5, help="best individuals copied unchanged into the next generation")
p.add_argument("--parents", type=positive_int, default=10, help="top N individuals allowed to have children (must be at least --elite)")
p.add_argument("--hidden", type=positive_int, default=12, help="neurons in the hidden layer")
p.add_argument("--games", type=positive_int, default=3, help="games each individual plays per generation")
p.add_argument("--seed", type=int, default=None, help="only for a new run (default 0)")
p.add_argument("--run", default="default", help="output folder name under runs/")
p.add_argument("--save-every", type=positive_int, default=5, help="save the whole generation every N generations")
p.add_argument("--verbose", action="store_true", help="print every individual's fitness each generation")
p.add_argument("--reset", action="store_true", help="delete this run and start over")
p.add_argument("-i", "--interactive", action="store_true",
               help="ask for each setting one by one instead of passing flags (cannot be combined with other options)")
a = p.parse_args()

if a.interactive:
    if sys.argv[1:] not in (["-i"], ["--interactive"]):
        p.error("--interactive asks for every setting itself; don't combine it with other options")
    try:
        answers = train_wizard(os.path.join(_venv.ROOT, "runs"))
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")
    if answers is None:
        sys.exit("Cancelled.")
    a = p.parse_args(answers)   # the same checks as typing the flags yourself

if os.sep in a.run or a.run in ("", ".", ".."):
    p.error("--run must be a plain folder name")
run_dir = os.path.join(_venv.ROOT, "runs", a.run)
if a.reset:
    shutil.rmtree(run_dir, ignore_errors=True)

try:
    cfg = Config(pop_size=a.pop, elite=a.elite, n_parents=a.parents, hidden=a.hidden, games_per_eval=a.games)
except ValueError as e:
    p.error(str(e))
train(cfg, a.generations, run_dir, seed=a.seed, save_every=a.save_every, verbose=a.verbose)
