# Jump-ML

A simple 2D dodge game, built so we can learn machine learning by teaching a bot to play it.

A character stands fixed in the middle of the screen. Obstacles slide in from the right:

- **Ground obstacle** (orange) → jump with `W` / `↑`
- **Air obstacle** (blue) → duck with `S` / `↓`

Each dodged obstacle scores **1–5 points**: faster and more tightly-spaced obstacles are worth more. A jump or duck that nothing needed **costs 1–5 points**, so mashing the keys doesn't pay. Score is shown top-right.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Play

```bash
scripts/play.py
```

Both launchers work from any directory and use the `.venv` automatically. `SPACE` restarts after a crash, `Esc` quits. Sound is generated in code; if no audio device exists the game runs silently.

## Train and watch a bot (Round 1)

```bash
scripts/train.py --generations 100
scripts/watch.py --brain runs/default
scripts/watch.py --brain perfect
scripts/train.py -i      # or: scripts/watch.py -i   (asks for each setting, shows the equivalent command)
scripts/watch.py --brain runs/default --multiple 0 25 50 95 --size 2 2   # compare generations
```

See `docs/03-neuroevolution.md` for how it works and what to expect.

## Test

```bash
scripts/run_tests.py
```

## Layout

Each folder below has its own `README.md`.

| Folder | What it is |
| --- | --- |
| `game_engine/` | The rules. Pure Python, no graphics or sound, so AIs can run it headless and fast. |
| `game_interface/` | The pygame window, drawing, keyboard and sound. |
| `docs/` | Game design and the ML learning guide. Start with `docs/02-ml-roadmap.md`; `docs/04-brain-and-breeding.md` explains the neural network and the breeding with diagrams. |
| `ml/` | The brains: neural network, perfect bot, and the neuroevolution trainer. |
| `scripts/` | Executable launchers: `play.py`, `train.py`, `watch.py`, `run_tests.py`. |
| `tests/` | Engine tests. |
| `runs/` | Training output (weights, CSV logs). Git-ignored. |

## Roadmap

1. Game engine + playable window (done)
2. Neuroevolution: a population of tiny neural nets that evolves over rounds (done)
3. Reinforcement learning (DQN)
4. An LLM as a coach / policy writer
