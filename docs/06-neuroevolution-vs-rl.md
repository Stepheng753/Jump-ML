# Neuroevolution vs reinforcement learning

Both methods train a neural network to play the same game with the same 10 inputs and 3 outputs. They differ in **how many networks they keep** and **when they learn**.

| | Neuroevolution (Round 1) | Reinforcement learning, DQN (Round 2) |
| --- | --- | --- |
| Networks at once | **50 individuals** | **1 network** (plus a frozen copy used to build targets) |
| Unit of progress | a **generation** | a **round** |
| What one unit plays | 50 individuals × 3 games = **150 games** | **10 training games** + **3 test games** |
| When learning happens | **between generations**, after the games are scored | **during the games**, every 4 steps |
| What learning changes | keeps the best individuals and mixes/mutates them | nudges the one network's weights toward better predictions |
| What the outputs mean | scores that only need the right order | predicted points still to come (Q-values) |
| What is saved | every generation: all 50 individuals, ranked | a snapshot of the network every 5 rounds, plus `best.npz` |
| Learning signal | the final score of each whole game | the reward of each step, plus a guess about the future |
| Speed here | about 95 s for 100 generations | about 30 s for 100 rounds |

## Neuroevolution: many networks, one winner each round

Think of a breeding programme.

1. Start with 50 random networks.
2. Each one plays 3 games. Its fitness is its average score.
3. The best 5 are copied unchanged (elites). The best 10 are parents.
4. The other 45 children are made by mixing two parents' numbers and nudging some of them at random.
5. That is the next generation. Repeat.

Nothing changes inside a game. A network's weights are fixed while it plays, and they only change when the next generation is made. Learning is selection: good numbers survive and spread.

Strengths: simple, and every individual is judged on whole games, so it can't be fooled by one step. It also keeps 50 candidates alive, so it doesn't rely on one network staying good.

Weaknesses: most of the 150 games give only one number each (the fitness), so it learns slowly per game played. And a useful change in one individual only matters once it is selected.

## DQN: one network that learns while it plays

Think of a player who learns during every match.

1. The network looks at the situation and predicts how many points each action will earn.
2. It picks the best action (or a random one, sometimes).
3. The game moves on and gives back the reward for that step.
4. The step is stored in a memory (the replay buffer).
5. Every 4 steps, the network learns from a random batch of stored steps: it compares its prediction with a target built from the real reward plus its best estimate of the future, then backprop nudges its weights.

So the same network is changing all the time, mid-game. Every step counts as a small lesson, not just the final score.

Strengths: each step teaches something, so it needs fewer games. There's one network to watch, and its checkpoints are simply the same network at different times.

Weaknesses: it can swing. A network can get good, then forget. That's why it saves `best.npz`, and why a single round's score can't be trusted alone.

## Why "round" and "generation" aren't the same

A generation is a population's worth of games: 50 different networks each play 3 games. A round is one network's worth of games: 10 training games where it learns, then 3 test games. A round of DQN is about 13 games, while a generation of neuroevolution is 150.

Their numbers can't be compared directly. Compare them by **the same number of game steps**, or by **time**.

## Which one to use

- **Neuroevolution** is the easier one to understand. Start with it to see the basic idea of selection.
- **DQN** is more sample-efficient and learns from every step. It's the standard approach for games like this one.

Both reach roughly the perfect bot's level here (about 120 out of a possible 126 on test games, depending on the seed).

## Watching them side by side

Both kinds of player can be watched together on the same obstacles:

Run `ml/run/watch.py` and choose **VS**. You add players one at a time (the perfect bot, or a generation or round from any run), and each list shows the score of every saved choice.

Each cell is labelled with its method, run, and score: `evo default g99 #1 fit 129.3` is an evolution individual (fitness is its training score), and `dqn dqn best r61 eval 131.1` is a DQN network (eval is its test score).
