# ML roadmap: from zero to a bot that plays forever

No prior ML needed. Read this once, then we do each round hands-on.

## 1. The five words you need

| Word | Meaning in our game |
| --- | --- |
| **Agent** | The bot that plays. |
| **Observation** | The numbers it sees each frame (`Game.get_observation()`): where the next obstacles are, how fast, what type. |
| **Action** | What it does: nothing, jump, or duck. |
| **Reward / fitness** | A number saying how well it did. For us: the score. |
| **Policy** | The function `observation → action`. This is the "brain" we are trying to learn. |

Training means finding a policy that gets a high score.

## 2. The brain: a tiny neural network

A neural network here is just a math formula with adjustable numbers (**weights**):

```
10 observation numbers → [ 12 hidden numbers ] → 3 scores (nothing / jump / duck)
```

Each arrow is a multiply-and-add using weights. The bot picks the action with the highest score. At the start the weights are random, so the bot flails. **Learning means finding better weights.** The total is about 170 numbers, so it is tiny and fast.

## 3. The plan: three rounds

### Round 1: Neuroevolution (survival of the fittest)
Same idea as evolution:
1. Make **50 bots** with random weights (generation 0).
2. Let each play one game. Score = fitness.
3. **Keep the best ~10.**
4. Fill the population again with **mutated copies** (add small random noise to the weights) and some **crossovers** (mix the weights of two parents).
5. Repeat. Each repeat is a **generation**, and each generation is one "round" of training.

Over many generations the average score climbs. No calculus or reward design is needed, so it is the best first step. We save the best weights to `runs/` and then watch the champion play in the real window.

### Round 2: Reinforcement learning (DQN)
Instead of a population, **one** network learns *during* play:
- After every frame it records (what it saw, what it did, reward, what happened next).
- It trains on random batches of those memories (a **replay buffer**).
- It learns to predict "how much total future score will I get if I take this action now?" (the **Q-value**) and picks the best.
- It sometimes tries random actions (**exploration**) so it can discover better moves.

This is the standard way modern game AIs learn. We then compare its learning curve with Round 1.

### Round 3: Adding an LLM
An LLM like Claude is far too slow to choose an action 60 times a second, and it is the wrong tool for reflexes. It is good at **thinking about the training**, so we give it two jobs:
- **Coach:** it reads the training logs ("generation 40: best 120, avg 30, bots die on air obstacles") and suggests changes: mutation size, reward tweaks, what to try next.
- **Policy writer:** it writes a small rule-based Python policy ("if an air obstacle is closer than X, duck"). We test the policy in the engine, feed the results back, and it improves the code over several rounds.

We can then compare all three approaches on the same seeds.

## 4. "Play indefinitely"
Once a bot is good, we make the game harder over time (**curriculum**: wider speed range, tighter gaps) and measure survival across many random seeds. A bot that survives 10+ minutes on 100 different seeds counts as "indefinitely" for our purposes. Saved checkpoints let us stop and continue training any time.

## 5. Where data lives
No database. Small files in `runs/<run name>/`: `gen_XXXX.npz` (a saved champion's weights, every 10 generations), `history.csv` (best and average score per generation), and later RL checkpoints.

## 6. Why the engine is separate from the window
Training plays millions of frames. The engine runs about 200,000 steps/second because it never draws or plays sound. The pygame window is only used to *watch* a trained bot.

## Next step
Round 1 is built. See `docs/03-neuroevolution.md`.
