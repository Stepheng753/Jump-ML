# Round 1: neuroevolution

Everything here is the code in `ml/`. Read it alongside this page.

## The brain (`ml/network.py`)

```
10 inputs → 12 hidden (tanh) → 3 outputs (raw scores) → pick the highest
```

| Inputs | What they are |
| --- | --- |
| 0, 1, 2 | Player state as three 0/1 flags: running, jumping, ducking |
| 3 | Progress through the current jump or duck (0..1) |
| 4, 5, 6 | Nearest obstacle: distance (1 = just spawned, 0 = at you), speed (0..1), type (0 ground, 1 air) |
| 7, 8, 9 | The same three numbers for the second-nearest obstacle |

Same forward pass as your MNIST notebook, `A = activation(W @ A_prev + B)`, with three differences:
- tanh on the hidden layer instead of sigmoid;
- raw output scores with no activation (we only take the argmax);
- **no backprop.** All 171 weights and biases (10×12+12 + 12×3+3) sit in one flat list, the **genome**.

## One generation (`ml/neuroevolution/evolve.py`)

1. **Play.** Each of the 50 bots plays 3 games. Every bot gets the same 3 seeds, so the comparison is fair. The seeds change every generation so a bot can't memorize one level.
2. **Rank.** Fitness = average score over those games. Games stop at 1 minute (3600 steps) so strong bots don't play forever.
3. **Breed.**
   - The best 5 are copied unchanged (**elitism**, so progress is never lost).
   - The other 45 are children of the top 10 (the **parents**; this includes the 5 elites, who both survive unchanged *and* can have children). With 70% chance a child **crosses over** two parents (each number comes from one or the other), otherwise it copies one parent.
   - Then **mutation**: each number has a 15% chance of a random nudge (standard deviation 0.3).

Generation 0 is just the random population.

## Run it

`--generations N` means "run N **more** generations". The whole population is saved to `runs/<run>/state.npz` after every call, so the next call carries on from there.

```bash
scripts/train.py --generations 1 --verbose    # one generation, shows every bot's fitness
scripts/train.py --generations 10             # 10 more
scripts/train.py --generations 50 --reset     # delete the run and start over
```

Or let it ask you for each setting: `scripts/train.py -i` (generations default to 100). It prints the equivalent command before starting.

Output goes to `runs/default/` (change with `--run NAME`):

| File | Contents |
| --- | --- |
| `history.csv` | best and mean fitness per generation |
| `gen_XXXX.npz` | one **whole generation**: all 50 individuals ranked best-first (50 × 171 numbers), their fitness, and the 3 game seeds they played. Saved every `--save-every` generations (default 5: gens 0, 5, 10, ...) and after the last generation of each call |
| `state.npz` | the population waiting to play the next generation (this is what lets training resume) |

One individual is **171 numbers** (its weights and biases). A `gen_XXXX.npz` holds 50 of them.

Other options: `--pop`, `--hidden`, `--games`, `--seed` (new runs only), and the breeding settings:

| Option | Default | Meaning |
| --- | --- | --- |
| `--elite N` | 5 | the top N individuals are copied unchanged into the next generation |
| `--parents N` | 10 | the top N individuals are allowed to have children. Must be at least `--elite` (the elites are part of the parent pool), otherwise training refuses to start |

The other `--pop - --elite` slots in each generation are children.

Then watch bots play in the real window:

```bash
scripts/watch.py --brain perfect
scripts/watch.py --brain runs/default/gen_0005.npz
scripts/watch.py --brain runs/default                  # latest saved generation
scripts/watch.py --brain runs/default --rank 3         # that generation's 3rd best individual
scripts/watch.py --brain runs/default --speed 8        # faster playback (any number, e.g. 20)
```

By default every game uses a **fresh random seed**, so you see how well the individual really plays, not how it did on the 3 games it was picked for. The top of the window shows `gen 40 | individual #1 | train fit 17.3 | game 2 | seed 481516916 | 1x`; the `train fit` is its score on *other* games, so it won't match. Options:

```bash
scripts/watch.py --brain runs/default --seed 123      # one specific obstacle sequence
scripts/watch.py --brain runs/default --replay        # the 3 games it was scored on in training
```

`--replay` shows `fitness 17.3 | game 2/3 | seed 4002` and reproduces the saved fitness exactly. Files saved before seeds were recorded can't be replayed.

Keys: `P` pause, `N` one frame while paused, `[` and `]` slower/faster, `SPACE` skip to the next game, `Esc` quit. The ground strip shows the net's three output scores and which one won.

`scripts/watch.py -i` asks what to watch step by step, in the same way.

### Comparing generations side by side

```bash
scripts/watch.py --brain runs/default --multiple 0 25 50 95                  # 1 row of 4
scripts/watch.py --brain runs/default --multiple 0 25 50 95 --size 2 2       # 2 rows x 2 columns
scripts/watch.py --brain runs/default --multiple 0 25 50 95 --size 4 1       # 4 rows x 1 column
```

- **Same game for everyone.** Every cell plays the same obstacles (a shared fresh seed), so any difference is the brain, not luck. When all cells have crashed, the next round starts with a new shared seed. Pass `--seed N` to choose it.
- **Generation order.** Cells run left to right, then top to bottom: the earliest generation is top-left and the latest is bottom-right, whatever order you typed them in. Compare the `fit` numbers to see whether later really is better; it isn't always.
- **What a cell shows.** `gen 25 | #1 | fit 17.3` is the generation, the individual's rank in it (`--rank` applies to every cell), and its training fitness. The big number is its live score this round, and the cell with the top live score gets a green border. `fit` is from training games, so it won't match a single round's score.
- **`--size ROWS COLS`.** `ROWS x COLS` must equal the number of generations, otherwise it stops with an error. With no `--size` you get one row.
- **Only saved generations work.** Training saves every `--save-every` generations (default 5), so asking for generation 1 fails and lists what exists. Train with `--save-every 1` to be able to show any generation.
- No sound in the grid, and the keys are the same (`SPACE` skips to the next round).

## What we saw (an earlier 100-generation run, seed 0; its files have since been deleted)

Champions tested on 20 games with seeds never used in training:

| Brain | Average score | Survived the full minute |
| --- | --- | --- |
| generation 0 (random) | 1.6 | 0 / 20 |
| generation 50 | 14.9 | 0 / 20 |
| generation 60 | 112.8 | 15 / 20 |
| generation 100 | 120.0 | 19 / 20 |
| perfect bot | 124.8 | 20 / 20 |

Notice the jump between generations 50 and 60. Evolution is not smooth: it plateaus for a while, then one lucky mutation lets bots handle both obstacle types and the score leaps.

## Things to know

- **Scores from before the wasted-action penalty aren't comparable.** Fitness is the game score, and the score now includes penalties for needless jumps and ducks. A run trained under the old rules shows fitness numbers (and in `--replay`, scores) from the old rules, while `watch` plays the new ones. Start a new run (`--run NAME`) to train under the current rules.
- **"Best" can go down between generations.** The 5 elites are copied unchanged, but each generation uses new game seeds, so the same bot can score lower on easier or harder games.
- **"Best" in the training log is noisy.** It is the top of 50 bots, each judged on only 3 games, so some of that number is luck. That is why we re-test champions on unseen seeds.
- **A score of ~120 is the ceiling here**, because each game is capped at one minute and the perfect bot hits it too. To measure "plays indefinitely" we need longer games and harder levels (a later step).
- **Why the net looks idle during a jump.** Once a jump or duck starts, the engine ignores new actions until it finishes, so the net's scores don't matter mid-air.
- **Experiments to try:** `--hidden 8` vs `--hidden 24`, `--pop 100`, a different `--seed`. Compare the `history.csv` curves.

For diagrams of the network and the breeding step, and a worked example with real numbers, see `docs/04-brain-and-breeding.md`.
