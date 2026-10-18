# ml/run

The two commands you use to train and watch players, the saved settings they can read, and the question machinery behind them.

| File | What it is |
| --- | --- |
| `train.py` | train a player: asks neuroevolution or DQN first, then that kind's settings |
| `watch.py` | watch players in four ways: single, single VS, multiple, multiple VS |
| `default.json` | saved settings for both: a `train` part and a `watch` part |
| `_interactive.py` | the question-and-answer wizards. They ask the questions and return a *plan* (a dictionary of what to do) |
| `_defaults.py` | reads `default.json`: checks every value with the same rules as the questions, reports all problems at once, and returns the same kind of plan |
| `_args.py` | the answer checks the questions use (`positive_int`, `non_negative_int`, `positive_float`, `gamma_value`, `playback_speed`, `any_int`, `run_name`), so a bad answer such as a speed of -2 is explained instead of breaking the run |

A plan is carried out by `ml/plans.py` (training) or `game_interface/plan.py` (watching).

Run them like any script. Each is an executable file with a shebang, works from any directory, and uses the project's `.venv` automatically:

```bash
ml/run/train.py
ml/run/watch.py
```

**Neither takes arguments.** They ask you questions: press Enter to accept the default in `[brackets]`, a bad answer is explained and asked again, and before anything starts you see a summary of your settings and a `Start? [Y/n]`. Ctrl-C or Ctrl-D cancels. (`--help` still works, but only describes the script.)

## train.py

The first question is the kind of training. Then you choose: **use the saved settings** in `default.json` for that kind, or **answer the questions**. The kind comes first, so the saved file never decides it for you. The kinds:

1. **Neuroevolution**: a population of networks plays, the best are kept and bred. See `docs/03-neuroevolution.md`.
2. **Reinforcement learning (DQN)**: one network learns while it plays. See `docs/05-reinforcement-learning.md`. The two are compared in `docs/06-neuroevolution-vs-rl.md`.

Then it asks that kind's settings:

| | Neuroevolution asks | DQN asks |
| --- | --- | --- |
| Always | run name (`default`), generations to train (100), population (50), hidden neurons (12), elites (5), parents (10), games per individual (3), save every N generations (5), seed (0), print every individual's fitness (no) | run name (`dqn`), rounds to train (100), training games per round (10), hidden neurons (128), save every N rounds (5), seed (0) |
| Optional | none | "advanced settings" (default no): discount, learning rate, batch size, memory size, game steps per training step, target refresh, exploration length, test games per round |

Notes:

- **Existing runs.** If the run name already exists it says how much it has trained, then asks whether to **delete it and start over** (the default, with a warning of what will be deleted) or, if you answer `n`, **carry on** from where it stopped. When carrying on, settings that can't change (population, hidden neurons, DQN memory size, seed) aren't asked.
- **One kind per run.** A run folder belongs to the kind of training that made it. Each kind refuses a name the other one is using, and says so.
- **Results** go to `runs/<run name>/`: `history.csv`, `gen_XXXX.npz` checkpoints, `state.npz`, and for DQN `best.npz`. The state needed to carry on is saved when a run finishes. If you interrupt one with Ctrl-C, the checkpoints and history written so far are kept, but the next run carries on from where the last *finished* run stopped, so its history file will repeat some rows.
- Training runs headless (no window) and prints one line per generation or round.

## watch.py

The first question is the **type of watching**:

1. **single**: one player. It first asks the kind (an evolution player, a DQN player, or the perfect bot), then the run and the checkpoint.
2. **single VS**: an evolution player on the **left** against a DQN player on the **right**, on the same obstacles.
3. **multiple**: several generations (or rounds) of **one** kind of training side by side. It first asks the kind (evolution or DQN), then the run, which generations, and the grid shape. When you type the list you can use `latest` (the highest saved) and, for a DQN run, `best`; the default for a DQN run ends with `best`.
4. **multiple VS**: evolution generations on the **left** and DQN rounds on the **right**, one pair per row. The grid is `rows x 2`, both sides list the same number of checkpoints, and each row pairs one of each. The default is a 3 x 2 grid: evolution 0, 50, 99 against DQN 0, 99, best.

Then, as in `train.py`, you choose **the saved settings** for that type or **the questions** (see "Saved settings" below). So for single and multiple the order is: type, then which kind, then saved settings or questions. The VS types have no kind to choose (they always pair evolution with DQN), and the perfect bot skips the saved-settings choice. **The file never decides neuroevolution vs DQN; you do, every time.** If a kind has no saved runs (for example there is no DQN run for a VS), it tells you so and stops.

Wherever it asks which generation or round, it first lists every saved one with its score (**best** = a generation's best individual's training fitness; **eval** = a DQN round's test score), so you can choose by result. For a DQN run it says "round", skips the "which individual" question (there is only one network) and offers **best** (`best.npz`) as the default.

Other questions: the game seed (Enter for random), whether to replay the games a generation was scored on (single only), and the playback speed (any number above 0, for example 0.5, 2, 8, 20).

Keys in the window: `P` pause, `N` one frame while paused, `[` and `]` slower/faster, `SPACE` skip to the next game (next round in a grid), `Esc` quit. In a grid every cell plays the same obstacles, cells are labelled with their score, and the cell with the top live score gets a green border. Single and multiple order cells by generation; VS cells fill row by row (left, right, left, right...) and say which kind they are (`evo evolution g50 #1 fit 17.3`, `dqn dqn best r64 eval 128.2`).

## Saved settings: default.json

`train.py` asks the kind of training first, then:

```
How do you want to set this up?
  1) Answer the questions
  2) Use the saved neuroevolution settings in default.json
```

(The menu names the kind you picked; Enter takes option 2.) `watch.py` asks the type of watching first (and, for single and multiple, the kind), then the same question. Choose the saved settings to skip the questions and use `ml/run/default.json`. You still get the summary and `Start? [Y/n]`, so nothing starts, and nothing is deleted, without your go-ahead. Edit the file in any text editor to change the settings; the shipped values are the defaults of the questions. Keys starting with an underscore (like `_note`) are ignored, an unknown key (a typo) is reported, and if anything is wrong you get **every** problem listed, naming the setting (`train.dqn.gamma`, `watch.vs.players[1]`), and then the questions instead.

The file has two parts, `"train"` (read by `train.py`) and `"watch"` (read by `watch.py`); each script ignores the other's part. **Neither part says which kind to use**: each has an `evolution` and a `dqn` section, and the one you pick when the script runs is the one that is read. A leftover `method` or `training` key is reported as "not used".

**The `train` part.** Its two sections are `evolution` and `dqn`; you choose the kind when the script runs, and only that section is used.

| Setting | Meaning |
| --- | --- |
| `run` | the folder under `runs/` |
| `existing_run` | `"continue"` adds to a run that already exists; `"start_over"` deletes it first (the summary warns you). When continuing, `population` / `hidden` / `memory_size` must match the run, and `seed` is ignored |
| evolution: `generations`, `population`, `elites`, `parents`, `hidden`, `games_per_individual`, `save_every`, `seed`, `verbose` | the same as the questions; `parents` must be at least `elites` |
| dqn: `rounds`, `training_games_per_round`, `hidden`, `save_every`, `seed`, `gamma`, `learning_rate`, `batch_size`, `memory_size`, `train_every`, `target_sync`, `exploration_steps`, `test_games_per_round` | the same as the questions, including the advanced ones |

The shipped file has `"existing_run": "continue"` in both sections.

**The `watch` part.** `speed` is any number above 0 and `seed` is `null` for new random obstacles. The type of watching is chosen when the script runs, and only that type's section is used. A checkpoint is `"latest"`, `"best"` (DQN runs only) or a saved generation/round number.

| Section | Settings |
| --- | --- |
| `single` | `evolution` and `dqn`, each with a `run` and a `checkpoint`; then `individual` (1 = best) and `replay` (`true` replays the games it was scored on, so `seed` must be `null`). The perfect bot needs no settings |
| `single_vs` | `evolution` and `dqn`, each with a `run` and a `checkpoint` (evolution on the left, DQN on the right) |
| `multiple` | `evolution` and `dqn`, each with a `run` and a `generations` list; then `individual`, `rows`, `cols` (both `null` for one row). A list holds saved numbers, `"latest"` (the highest saved number, either kind) and `"best"` (the best checkpoint, DQN runs only), for example `[0, 50, "latest"]` or `[0, 99, "best"]`. Numbers are shown earliest first (top-left) and `"best"` always comes last, because it is a checkpoint, not a point in time |
| `multiple_vs` | `evolution` and `dqn`, each with a `run` and a `generations` list (`"best"` allowed on the DQN side). Both lists need the same length, and each row pairs one of each. `rows` must be that length and `cols` must be 2 (or both `null`) |

Each side must be the right kind of run: an evolution side pointing at a DQN run is reported. The shipped file is a 3 x 2 multiple VS of the run `evolution` (0, 50, 99) against the run `dqn` (0, 99, best), and its other sections name the same two runs, so it needs those runs to exist.

## Tests

`tests/test_interactive.py` (the questions), `tests/test_defaults.py` (the settings file), `tests/test_plans.py` (carrying out the answers) and `tests/test_args.py` (the answer checks).
