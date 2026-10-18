# Reinforcement learning (DQN)

In this method **one network** learns while it plays, from the reward of every step. There is no population. Compare with neuroevolution in `docs/06-neuroevolution-vs-rl.md`.

Code: `ml/rl/` (`replay.py` the memory, `agent.py` the learning math, `train.py` the loop). Run it with `ml/run/train.py` and choose DQN.

## The idea in one paragraph

At every step the network looks at the situation, predicts how many points each action will earn, and picks the best one. The game then tells us what really happened. We compare the prediction with a better estimate made from that real result, and nudge the weights so the next prediction is a little closer. Repeat for hundreds of thousands of steps.

## Predictions, not flags

The network has three outputs, one per action:

- `Q(nothing)`: how many more points I expect if I do nothing now.
- `Q(jump)`: the same, if I jump now.
- `Q(duck)`: the same, if I duck now.

Picking the biggest is the action. The number is a prediction, not an on/off flag.

## One step, with numbers

We're on frame 50. A ground obstacle is coming and the player is running.

1. **Predict.** The network looks at frame 50 and predicts `Q(jump) = 1`, so it jumps. We predicted 1 point for jumping.
2. **What really happened.** Frame 50 earned 0 points, because the jump didn't pass anything yet. That's the reward `r = 0`.
3. **Look ahead one frame.** We're on frame 51. The network looks at frame 51 and says the best it can do from here is `Q = 2` (for doing nothing). That's its estimate of the points still to come.
4. **Build the target.** The value frame 50 *should* have predicted is:

   ```
   target = reward + gamma × best next estimate
          = 0 + 0.9 × 2
          = 1.8
   ```

   Gamma (0.9 here) makes points later count a little less than points now.

5. **Find the error.** Compare the target with the prediction:

   ```
   error = target − prediction = 1.8 − 1 = 0.8
   ```

6. **Learn.** Backprop uses the error to change the weights and biases, so that next time the network sees a situation like frame 50, its jump prediction is a bit closer to 1.8. Only the jump output is corrected, because only jump was taken.

Two things to notice:

- The **target** is part real (the 0 reward) and part guess (the 2 from frame 51). The real reward is what keeps the guesses honest over time.
- The update **uses frame 50**, the step that just happened. Frame 51's own error comes later, from frame 51 and frame 52.

## Why not only learn from the step that just happened?

We can learn from any stored step. The memory of recent steps is called the **replay buffer**. Each step is saved as one record: `(situation, action, reward, next situation)`. Every 4 game steps, the trainer draws 64 records at random from the last 100,000 and learns from all of them.

So an update might use frame 50 together with frame 40 and frame 12. Using random old steps:

- avoids learning only from the last few frames, which look almost identical to each other;
- lets a single good or bad experience be used many times.

When an old record is reused, its error is recalculated with the network's **current** weights, so it always reflects what the network knows now.

## Learning and playing happen together

There is no "learn at the end". The loop for each step is:

1. predict and pick an action (sometimes a random one, see below);
2. play the step and save the record;
3. every 4 steps, learn from a random batch.

The weights change continuously, in the middle of games.

## Exploration: eps

Early on, the predictions are close to random, so following them only teaches random habits. So for some steps the agent picks a **random action** instead of the best one. The chance of that is **eps**.

- eps starts at 1.00 (every action random)
- and falls in a straight line with the number of steps
- to 0.05 after 50,000 steps, then stays there.

Some random actions remain to the end, so the agent keeps finding out about alternatives.

## Rounds and what gets printed

A **round** is a block of work that gets one summary line:

- **10 training games** with random actions allowed (the learning happens inside these games);
- then **3 test games** with no random actions and no learning. The test uses the same fixed games every round, so rounds can be compared.

```
round    1 | eval     0.0 | train   -14.1 | eps 0.93 | loss   0.095 | steps this round 1,538 | total 3,507
```

| Field | Meaning |
| --- | --- |
| `round 1` | counting starts at 0, so this is the second round |
| `eval 0.0` | average score of the 3 test games. The number that matters |
| `train −14.1` | average score of the 10 training games, with random actions. Negative means points lost to wasted presses and crashes outweighed points earned |
| `eps 0.93` | the random-action chance after this round |
| `loss 0.095` | average size of the errors learned from this round. It shows the predictions are settling, not how well the bot plays |
| `steps this round 1,538` | game steps played in this round, training games only |
| `total 3,507` | game steps since training began |

## Keeping the best network

DQN swings from round to round. One run first scored above 100 at round 32, then averaged only 36 over its next ten rounds. So the **last round is not reliably the best**.

Whenever a round scores at least as well as the best so far, its network is re-tested on 10 fresh games. If it still holds up, it's saved as `best.npz`. Watch that file:

Choose `best` when `ml/run/watch.py` asks which round.

## Run it

```bash
ml/run/train.py
```

It asks which kind of training; choose **2, reinforcement learning (DQN)**. Press Enter to accept a default in `[brackets]`. Training 100 rounds takes about 30 seconds on this machine. On an existing run, the number of rounds means that many **more**, and it carries on where it stopped (or you can delete the run and start over, which is the default).

Then watch it. **Watch `best.npz`, not the last round** (see above): run `ml/run/watch.py` and choose the run; `best` is the default choice, and every round is listed with its test score first.

Output goes to `runs/dqn/` (or the run name you chose):

| File | What it is |
| --- | --- |
| `history.csv` | one line per round, the same numbers as the printed line |
| `gen_XXXX.npz` | a snapshot of the network every 5 rounds (weights and biases only) |
| `best.npz` | the best network so far |
| `state.npz` | everything needed to carry on: weights, the frozen copy, Adam's memory, the replay buffer |

## Settings

The first questions are the run name, how many rounds, training games per round (10), hidden neurons (128), how often to save (5) and the seed (0). The **advanced settings** are optional (the question defaults to no):

| Setting | Default | What it does |
| --- | --- | --- |
| Discount (gamma) | 0.99 | how much a point one step later counts compared with one now |
| Learning rate | 0.001 | how big each weight change is |
| Batch size | 64 | steps per learning update |
| Memory | 100,000 | how many recent steps the replay buffer keeps (fixed once a run exists) |
| Game steps per training step | 4 | how often it learns |
| Target refresh | 4000 | steps between refreshing the frozen copy used to build targets |
| Exploration | 50,000 | steps for eps to fall from 1.00 to 0.05 |
| Test games per round | 3 | the test that gives the eval score |

Hidden neurons are also fixed once a run exists; the other settings can change each time you carry on.

## What we measured

Test games use seeds never used for training. The perfect bot averages about 126 on them.

- With the first defaults, one seed reached a strong network and then forgot it; only 3 of 6 new seeds were still good at the end.
- Keeping `best.npz` fixed most of that: on six unseen seeds, `best.npz` averaged 115 and was good on all six. The last round averaged 109 on the same seeds.
- Changing the learning rate, exploration floor, buffer size or target refresh alone did not help consistently. A bigger network helped a little.

Small samples (4 to 6 seeds, one game each), so treat these as guides.

## Things it does not do yet

- **Several steps ahead.** This version looks one step ahead, and that is enough because the next step's estimate already includes everything after it. Looking several real steps ahead (n-step) is a common improvement.
- **Smarter sampling.** Random batches treat every step equally. Prioritized replay would learn more from surprising steps.
