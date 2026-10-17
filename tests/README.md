# tests

Run everything from the project root:

```bash
scripts/run_tests.py            # whole suite
scripts/run_tests.py -v         # list each test
scripts/run_tests.py -k perfect # only tests with "perfect" in the name
```

Extra arguments go straight to pytest. `pytest.ini` at the project root puts the project folder and `scripts/` on the import path.

## What each file checks

| File | Covers |
| --- | --- |
| `test_engine.py` | the game rules. The engine imports no pygame; points always fall between 1 and 5 and rise with speed and closeness; spawn gaps respect the minimum; doing nothing eventually dies; a perfect-timing bot survives 2 minutes on 30 seeds (so the spawner never makes an impossible pattern); same seed gives the same game; the observation is 10 numbers with one-hot state; blue obstacles can't be jumped over; the wasted-action penalty (range, direction, which presses count as needed, the perfect bot never wastes one) |
| `test_ml.py` | the learning code. Layout is 10 → 12 → 3 with 171 numbers; the net outputs 3 scores; the perfect brain survives; crossover and mutation behave; elites are preserved; save and load round trip; a short training run writes the right files; resuming matches one long run; `--parents` must be at least `--elite` |
| `test_grid.py` | the multi-generation view's logic: `--size` must match the number of generations, cells are ordered by generation, the layout fits the window, and asking for an unsaved generation lists the saved ones |
| `test_interactive.py` | the `-i` wizards, driven by scripted answers: defaults, re-asking on bad input (parents below elites, unsaved generations, wrong grid shape), continuing vs resetting an existing run, and every watch path (one generation, replay, compare several, perfect bot) |
| `test_args.py` | the argument checks: zero, negative, `nan`, `inf` and non-numbers are rejected for `--speed` and the whole-number options |

## Things to know

- **No test imports pygame.** `test_engine_has_no_pygame` checks that the engine stays free of it, and pytest imports every test file before running any, so a pygame import anywhere would make that test fail. That is why the grid maths lives in `game_interface/grid.py`, which has no pygame import. The windows themselves are checked by running them by hand.
- Training tests write to a temporary folder, never to your real `runs/`.
