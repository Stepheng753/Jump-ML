#!/usr/bin/env python3
"""Watch an individual (or the perfect bot) play in the real game window:

    scripts/watch.py --brain perfect
    scripts/watch.py --brain runs/default/gen_0040.npz
    scripts/watch.py --brain runs/default          (latest saved generation, best individual)
    scripts/watch.py --brain runs/default --rank 3 (that generation's 3rd best individual)
    scripts/watch.py --brain runs/default --speed 4
    scripts/watch.py --brain runs/default --seed 123      (one specific obstacle sequence)
    scripts/watch.py --brain runs/default --replay        (the 3 games it was scored on in training)

Compare several generations side by side (they all play the same obstacles):

    scripts/watch.py --brain runs/default --multiple 0 25 50 95
    scripts/watch.py --brain runs/default --multiple 0 25 50 95 --size 2 2
    scripts/watch.py --brain runs/default --multiple 0 25 50 95 --size 4 1

Cells are ordered by generation, left to right then top to bottom: the earliest
generation is top-left and the latest is bottom-right, whatever order you typed
them in. Only generations that were saved can be shown; training saves every
--save-every generations (default 5).

By default every game uses a fresh random seed, so you see how well the individual
really plays. The fitness in the label is its training score on other games.
--replay plays the exact games it was scored on instead (game 1/3, 2/3, 3/3).

Keys: P pause, N step one frame (while paused), [ / ] slower / faster,
SPACE skip to the next game (next round in the grid), Esc quit.

Not sure which flags you want? Let it ask you:

    scripts/watch.py -i
"""
import argparse
import os
import sys

import _venv
from _args import playback_speed, positive_int
from _interactive import watch_wizard

_venv.ensure(__file__)
from game_interface.grid import by_generation, resolve_size  # noqa: E402
from ml.brains import load_brain, load_generation  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--brain", help="'perfect', a .npz file, or a run folder (required unless --interactive)")
p.add_argument("--rank", type=positive_int, default=1, help="which individual of the saved generation (1 = best)")
p.add_argument("--multiple", type=int, nargs="+", metavar="GEN",
               help="generations to show side by side, e.g. --multiple 0 25 50 95 (--brain must be a run folder)")
p.add_argument("--size", type=int, nargs=2, metavar=("ROWS", "COLS"),
               help="grid shape for --multiple, e.g. --size 2 2. ROWS x COLS must equal the number of generations. "
                    "Default: 1 row")
p.add_argument("--seed", type=int, default=None,
               help="start from this game seed (default: random). The grid uses one shared seed for all cells")
p.add_argument("--replay", action="store_true",
               help="play the games the individual was scored on in training instead of fresh ones")
p.add_argument("--speed", type=playback_speed, default=1.0, help="playback speed: any number above 0, up to 1000 (e.g. 0.5, 2, 8, 20)")
p.add_argument("-i", "--interactive", action="store_true",
               help="ask what to watch one question at a time instead of passing flags (cannot be combined with other options)")
a = p.parse_args()

if a.interactive:
    if sys.argv[1:] not in (["-i"], ["--interactive"]):
        p.error("--interactive asks for every setting itself; don't combine it with other options")
    try:
        answers = watch_wizard(os.path.join(_venv.ROOT, "runs"))
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")
    if answers is None:
        sys.exit("Cancelled.")
    a = p.parse_args(answers)   # the same checks as typing the flags yourself
if not a.brain:
    p.error("--brain is required (or use --interactive to be asked)")

if a.size and not a.multiple:
    p.error("--size only makes sense together with --multiple")
if a.replay and a.seed is not None:
    p.error("--replay and --seed conflict: --replay uses the saved training seeds")
if a.replay and a.multiple:
    p.error("--replay works on one individual; the grid always plays one shared fresh game")

if a.multiple:
    if not os.path.isdir(a.brain):
        p.error("--multiple needs --brain to be a run folder, like --brain runs/default")
    if len(set(a.multiple)) != len(a.multiple):
        p.error("--multiple lists the same generation more than once")
    try:
        rows, cols = resolve_size(len(a.multiple), a.size)
        brains = by_generation([load_generation(a.brain, g, a.rank) for g in a.multiple])
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    from game_interface.watcher import watch_grid  # noqa: E402

    watch_grid(brains, rows, cols, seed=a.seed, speed=a.speed)
else:
    from game_interface.watcher import watch  # noqa: E402

    try:
        brain = load_brain(a.brain, rank=a.rank)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    if a.replay and not getattr(brain, "seeds", None):
        p.error("--replay needs a saved generation that recorded its training games; this brain has none")
    watch(brain, seed=a.seed, speed=a.speed, replay=a.replay)
