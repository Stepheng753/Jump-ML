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

```bash
ml/run/train.py
```

It asks which kind of training; choose **1, neuroevolution**. Press Enter to accept a default in `[brackets]`. Then:

| Question | Default | Meaning |
| --- | --- | --- |
| Run name | `default` | the folder under `runs/` that holds the results |
| How many generations to train | 100 | generations to run now. On an existing run it means that many **more** |
| Population size | 50 | individuals per generation |
| Hidden neurons | 12 | size of each network's hidden layer |
| Elites | 5 | the top N are copied unchanged into the next generation |
| Parents | 10 | the top N are allowed to have children. Must be at least the elites (the elites are part of the parent pool); it will not accept less |
| Games each individual plays | 3 | per generation |
| Save every N generations | 5 | how often a whole generation is written to disk |
| Random seed | 0 | new runs only. The same seed and settings give the same run |
| Print every individual's fitness | no | one line per individual, per generation: useful for looking closely at 1 or 2 generations |

Then it shows your settings and asks `Start? [Y/n]`.

If the run name already exists it says how much it has trained and asks whether to **delete it and start over** (the default; it warns what will be deleted) or, if you answer `n`, **carry on** from where it stopped. When carrying on, the population size, hidden neurons and seed are fixed by the run, so it doesn't ask for them. Each run belongs to one kind of training: it will not reuse a name that a DQN run is using.

To look closely at one generation at a time, train 1 generation with "Print every individual's fitness" on, then run it again and answer `n` to the start-over question to add more.

Output goes to `runs/<run name>/`:

| File | Contents |
| --- | --- |
| `history.csv` | best and mean fitness per generation |
| `gen_XXXX.npz` | one **whole generation**: all 50 individuals ranked best-first (50 × 171 numbers), their fitness, and the 3 game seeds they played. Saved every N generations (default 5: gens 0, 5, 10, ...) and after the last generation of each run |
| `state.npz` | the population waiting to play the next generation (this is what lets training carry on) |

One individual is **171 numbers** (its weights and biases). A `gen_XXXX.npz` holds 50 of them. The other (population − elites) slots in each generation are children.

## Watch it

```bash
ml/run/watch.py
```

It asks what to watch; pick the run. Wherever it asks for a generation, it first lists every saved one with its **best** score (the generation's best individual's training fitness). Then you choose:

- **Which generation**, or latest, and **which individual** of it (1 = best, 2 = second best, ...).
- **New random games** (the default) or **replay** the exact games it was scored on. By default every game uses a fresh random seed, so you see how well the individual really plays, not how it did on the 3 games it was picked for. You can also choose a specific game seed.
- **Playback speed**: any number above 0 (for example 0.5, 2, 8, 20).

The top of the window shows `gen 40 | individual #1 | train fit 17.3 | game 2 | seed 481516916 | 1x`. The `train fit` is its score on *other* games, so it won't match. With replay it shows `fitness 17.3 | game 2/3 | seed 4002` and reproduces the saved fitness exactly. Files saved before seeds were recorded can't be replayed.

Keys: `P` pause, `N` one frame while paused, `[` and `]` slower/faster, `SPACE` skip to the next game, `Esc` quit. The ground strip shows the net's three output scores and which one won.

### Comparing generations side by side

Choose "several side by side" and type the generations, such as `0 25 50 95`, then the grid shape (rows and columns).

- **Same game for everyone.** Every cell plays the same obstacles (a shared fresh seed, or one you choose), so any difference is the brain, not luck. When all cells have crashed, the next round starts with a new shared seed.
- **Generation order.** Cells run left to right, then top to bottom: the earliest generation is top-left and the latest is bottom-right, whatever order you typed them in. Compare the `fit` numbers to see whether later really is better; it isn't always.
- **What a cell shows.** `gen 25 | #1 | fit 17.3` is the generation, the individual's rank in it, and its training fitness. The big number is its live score this round, and the cell with the top live score gets a green border. `fit` is from training games, so it won't match a single round's score.
- **The grid shape** must make exactly as many cells as generations, otherwise it asks again. The default is one row.
- **Only saved generations can be chosen.** Training saves every N generations (default 5), so generation 1 isn't offered unless you save every generation.
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
- **Experiments to try:** 8 vs 24 hidden neurons, a population of 100, a different seed. Compare the `history.csv` curves.

For diagrams of the network and the breeding step, and a worked example with real numbers, see `docs/04-brain-and-breeding.md`.
