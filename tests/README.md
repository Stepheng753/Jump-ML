# tests

Run everything from the project root:

```bash
scripts/run_tests.py
```

It takes no arguments. To run only some tests, call pytest directly, for example `.venv/bin/python -m pytest -k perfect`. `pytest.ini` at the project root puts the project folder, `scripts/` and `ml/run/` on the import path.

## What each file checks

| File | Covers |
| --- | --- |
| `test_engine.py` | the game rules. The engine imports no pygame; points always fall between 1 and 5 and rise with speed and closeness; spawn gaps respect the minimum; doing nothing eventually dies; a perfect-timing bot survives 2 minutes on 30 seeds (so the spawner never makes an impossible pattern); same seed gives the same game; the observation is 10 numbers with one-hot state; blue obstacles can't be jumped over; the wasted-action penalty (range, direction, which presses count as needed, the perfect bot never wastes one) |
| `test_dqn.py` | reinforcement learning: the hand-written backprop matches finite differences (and fails if deliberately broken), only the action taken gets an error, the target is `r + γ × best next Q` with no future after a crash, the target network stays frozen until synced, a toy problem is learned, exploration schedule, replay buffer, settings validation, resume equals one long run, the best-checkpoint rules, and the two methods refusing each other's runs |
| `test_labels.py` | the window labels, including the perfect bot (which once crashed the watcher) and DQN's "round" wording |
| `test_ml.py` | the learning code. Layout is 10 → 12 → 3 with 171 numbers; the net outputs 3 scores; the perfect brain survives; crossover and mutation behave; elites are preserved; save and load round trip; a short training run writes the right files; resuming matches one long run; `--parents` must be at least `--elite` |
| `test_grid.py` | the multi-generation view's logic: the grid shape must match the number of generations, cells are ordered by generation, the layout fits the window, and asking for an unsaved generation lists the saved ones |
| `test_interactive.py` | the question wizards, driven by scripted answers: the kind-of-training question, defaults, re-asking on bad input (parents below elites, unsaved generations, wrong counts), continuing vs deleting an existing run, each kind refusing the other's run names, and the four types of watching (generation lists can use `latest`, and `best` for DQN): single (evolution, DQN with `best` by default, the perfect bot), single VS (evolution left, DQN right), multiple, and multiple VS (3 x 2 with evolution on the left and DQN with `best` last on the right, both sides needing the same count), plus scores listed next to each choice |
| `test_defaults.py` | the saved settings file (`ml/run/default.json`): the shipped file holds the questions' defaults for both kinds and switching kind needs only `method`; both parts load from one file and each loader ignores the other's part; continuing an existing run needs the fixed settings to match; `start_over` and the other kind's run names; every problem reported at once with the setting named; missing and broken files; each watch mode; and the first question in both wizards (use the file, answer the questions, broken file falling back to the questions) |
| `test_plans.py` | carrying out the answers: a training plan runs the chosen method into its run folder, continues or resets it, refuses the other kind's run and bad run names; a watch plan loads the right players (generations ordered, VS order kept, labels) and refuses bad ones |
| `test_args.py` | the answer checks the questions use: zero, negative, `nan`, `inf` and non-numbers are rejected for speed, discount and the whole-number settings |

## Things to know

- **No test imports pygame.** `test_engine_has_no_pygame` checks that the engine stays free of it, and pytest imports every test file before running any, so a pygame import anywhere would make that test fail. That is why the grid maths lives in `game_interface/grid.py`, which has no pygame import. The windows themselves are checked by running them by hand.
- Training tests write to a temporary folder, never to your real `runs/`.
