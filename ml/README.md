# ml

The machine learning: the neural network, the players built from it, and the evolution that trains it. It uses `game_engine` directly and never opens a window, so training runs headless and fast. (Watching a trained brain is `game_interface`'s job.)

The idea in one line: a population of tiny neural networks plays the game, the best ones are kept and bred with small random changes, and that repeats for many generations. Diagrams and a worked example: `docs/04-brain-and-breeding.md`. Step-by-step run guide: `docs/03-neuroevolution.md`.

## Files

| File | What it holds |
| --- | --- |
| `network.py` | the brain itself: `Network`, plus `random_genome`, `genome_size`, `layer_sizes`. Default layout 10 inputs → 12 hidden (tanh) → 3 outputs, which is **171 numbers** in one flat list (the genome). There is no backprop |
| `brains.py` | the two kinds of player and the save/load code. `PerfectBrain` is a hand-written rule; `NetBrain` wraps a `Network` and picks the highest of its 3 scores. `save_generation`, `load_brain`, `load_generation`, `saved_generations` |
| `rollout.py` | `play_game(brain, seed, max_steps)` plays one headless game; `mean_score` averages several. Games stop at 3600 steps (1 game-minute) so strong brains don't play forever |
| `neuroevolution/evolve.py` | `Config` (population size, elites, parents, mutation settings), `next_generation` (elitism, crossover, mutation) and `train` (the loop, with save and resume) |

## The brain interface

Every player, hand-written or learned, looks the same, so trainer and watcher treat them identically:

```python
brain.act(observation) -> 0 (nothing) | 1 (jump) | 2 (duck)
```

To add a new kind of player (the later reinforcement-learning one, say), write a class with `name`, `reset()`, `act(obs)` and optionally `scores`, and it works with `play_game` and the watcher unchanged.

## What training writes (`runs/<name>/`)

| File | Contents |
| --- | --- |
| `gen_XXXX.npz` | one whole generation: `population` (all individuals, **best first**, shape 50 x 171), `fitness` for each, `sizes`, `generation`, and the `seeds` of the games they were scored on. Saved every `--save-every` generations (default 5) and after the last generation of each call |
| `state.npz` | the population waiting to play next, the next generation number and the run seed. This is what lets `train.py` resume |
| `history.csv` | best and mean fitness per generation |

Older single-champion `gen_*.npz` files (with a `genome` key) still load; they just have no other individuals and no saved seeds.

## Reproducibility and seeds

`--seed` is the run's master seed. From it come the starting population, the breeding randomness of each generation, and the training games (generation `g` plays seeds `seed x 1,000,000 + g x 100 + k`). The same seed with the same settings gives the same run, and resuming gives results identical to one long run (a test checks this).

## Tests

`tests/test_ml.py`: layout and genome size, crossover and mutation, elites preserved, save/load round trip, a short training run, resume equals one long run, and the parents >= elite rule.
