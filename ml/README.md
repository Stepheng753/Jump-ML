# ml

The machine learning: the neural network, the players built from it, and the two ways of training it (neuroevolution and DQN). It uses `game_engine` directly and never opens a window, so training runs headless and fast. (Watching a trained brain is `game_interface`'s job.)

Two ideas. **Neuroevolution:** a population of tiny neural networks plays the game, the best ones are kept and bred with small random changes, and that repeats for many generations (`docs/03-neuroevolution.md`, diagrams in `docs/04-brain-and-breeding.md`). **DQN:** one network learns while it plays, from the reward of every step (`docs/05-reinforcement-learning.md`).

## Files

| File | What it holds |
| --- | --- |
| `network.py` | the brain itself: `Network`, plus `random_genome`, `genome_size`, `layer_sizes`. Default layout 10 inputs → 12 hidden (tanh) → 3 outputs, which is **171 numbers** in one flat list (the genome). There is no backprop |
| `brains.py` | the two kinds of player and the save/load code. `PerfectBrain` is a hand-written rule; `NetBrain` wraps a `Network` and picks the highest of its 3 scores. `save_generation`, `load_brain`, `load_generation`, `saved_generations`, plus `load_spec` (one VS-mode player: `perfect`, a file, or `RUN:GEN`), `saved_scores` and `best_checkpoint` (what the wizard lists next to each choice) |
| `rollout.py` | `play_game(brain, seed, max_steps)` plays one headless game; `mean_score` averages several. Games stop at 3600 steps (1 game-minute) so strong brains don't play forever |
| `run/` | the commands `train.py` and `watch.py` that ask you questions (or read `run/default.json`), and the machinery behind them. See `run/README.md` |
| `plans.py` | `run_training(plan, runs_dir)`: carries out the training plan that the `ml/run/train.py` questions produce (which kind, run name, how many, the settings), including deleting the run first if you chose to start over |
| `runs.py` | `run_method(run_dir)`: was this folder made by `'evolution'` or `'dqn'`? Each method refuses to continue the other's run |
| `rl/replay.py` | `ReplayBuffer`: the memory of recent transitions `(s, a, r, s2, done)` |
| `rl/agent.py` | the DQN learner: `DQNAgent` (a live network and a frozen target copy), `td_loss_and_grad` (backprop written out by hand), `Adam` |
| `rl/train.py` | `DQNConfig` and `train`: the loop that plays, remembers, learns, tests each round, and keeps `best.npz`. See `docs/05-reinforcement-learning.md` |
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
| `gen_XXXX.npz` | one whole generation: `population` (all individuals, **best first**, shape 50 x 171), `fitness` for each, `sizes`, `generation`, and the `seeds` of the games they were scored on. Saved every N generations (you choose; default 5) and after the last generation of each run |
| `state.npz` | the population waiting to play next, the next generation number and the run seed. This is what lets `train.py` resume |
| `history.csv` | best and mean fitness per generation |

Older single-champion `gen_*.npz` files (with a `genome` key) still load; they just have no other individuals and no saved seeds.

## DQN runs

A DQN run folder holds `history.csv` (`round,eval_score,train_score,epsilon,loss,steps,round_steps`), `gen_XXXX.npz` checkpoints (one network each, saved with `kind="dqn"`, so the watcher says "round" instead of "generation"), `best.npz` (the best network so far, confirmed on 10 fresh games), and `state.npz` (the weights, the frozen copy, Adam's memory and the replay buffer, so resuming is exact). The checkpoint format is the same as an evolution generation holding a single individual, so `ml/run/watch.py` works on both.

## Reproducibility and seeds

The seed you choose is the run's master seed. From it come the starting population, the breeding randomness of each generation, and the training games (generation `g` plays seeds `seed x 1,000,000 + g x 100 + k`). The same seed with the same settings gives the same run, and resuming gives results identical to one long run (a test checks this).

## Tests

`tests/test_dqn.py` (backprop against finite differences, the target calculation, the replay buffer, a toy problem it must learn, resume equals one long run, best-checkpoint rules) and `tests/test_ml.py`: layout and genome size, crossover and mutation, elites preserved, save/load round trip, a short training run, resume equals one long run, and the parents >= elite rule.
