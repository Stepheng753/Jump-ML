# scripts

The commands you run. Each is an executable file with a shebang, so run it directly from the project root (no `python` needed). Each one supports `--help`.

| Script | What it does | Example |
| --- | --- | --- |
| `play.py` | play the game yourself | `scripts/play.py` |
| `train.py` | train by neuroevolution (headless, fast), resuming where it stopped | `scripts/train.py --generations 10 --run xyz` |
| `watch.py` | watch a brain play, or compare several generations side by side | `scripts/watch.py --brain runs/xyz --multiple 0 5 10 15 --size 2 2` |
| `run_tests.py` | run the test suite; extra arguments go to pytest | `scripts/run_tests.py -k perfect` |

## Interactive mode (`-i`)

Don't want to remember flags? `train.py` and `watch.py` can ask you instead:

```bash
scripts/train.py -i
scripts/watch.py -i
```

Each question shows its default in `[brackets]`; press Enter to accept it. Yes/no questions show the default first: `[N/y]` means the default is no, `[Y/n]` means yes. A bad answer is explained and asked again (for example, parents smaller than elites). At the end it prints the **equivalent command line** and asks `Start? [Y/n]`, so you learn the flags as you go. Ctrl-C or Ctrl-D cancels.

- **train** asks: run name, generations (**default 100**; the plain command's default is 10), population, hidden neurons, elites, parents, games per individual, save-every, seed, verbose. If the run name already exists it says so and offers to delete it and start over (**the default**; it warns what will be deleted before the final `Start?`) or to continue it (answer `n`; population, hidden size and seed are then fixed by the run, so it doesn't ask).
- **watch** asks: which run or the perfect bot (the most recently trained run is the default), one generation or several side by side, which generation(s), which individual, replay or fresh games, grid shape, seed, speed. It only offers generations that were actually saved.

`-i` can't be combined with other options; use one or the other.

## train.py in short

`--generations N` means N **more** generations; the population is saved after every call, so you can train 1 generation, look, then 10 more. Useful options:

```bash
scripts/train.py --generations 1 --verbose        # show every individual's fitness
scripts/train.py --generations 50 --reset         # wipe this run and start over
scripts/train.py --generations 20 --elite 2 --parents 8 --run small
```

Others: `--pop`, `--hidden`, `--games`, `--seed` (new runs only), `--save-every` (default 5), `--run NAME` (output folder under `runs/`). `--parents` must be at least `--elite`.

## watch.py in short

```bash
scripts/watch.py --brain perfect                               # the hand-written bot
scripts/watch.py --brain runs/xyz                              # latest saved generation, fresh random games
scripts/watch.py --brain runs/xyz --rank 3 --speed 4           # 3rd best individual, 4x speed
scripts/watch.py --brain runs/xyz --replay                     # the 3 games it was scored on in training
scripts/watch.py --brain runs/xyz --multiple 0 25 50 95 --size 2 2
```

The grid orders cells by generation (earliest top-left, latest bottom-right). Keys are listed in `scripts/watch.py --help`.

## Helper files (not commands)

| File | Purpose |
| --- | --- |
| `_venv.py` | makes each script re-run itself inside the project's `.venv` and puts the project root on the import path, so they work from any directory without activating anything |
| `_interactive.py` | the question-and-answer wizards behind `-i`. They ask, then return the equivalent flags, which the script parses with its normal argument parser, so both modes validate identically |
| `_args.py` | shared argument checks (`positive_int`, `non_negative_int`, `playback_speed`) so a bad value such as `--speed -2` stops with a clear message instead of breaking the run |

The leading underscore marks them as internal.
