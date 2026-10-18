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

Every script works from any directory and uses the `.venv` automatically. `SPACE` restarts after a crash, `Esc` quits. Sound is generated in code; if no audio device exists the game runs silently.

## Train and watch

Neither script takes arguments. Run it and answer the questions; press Enter to accept the default in `[brackets]`. `train.py` asks the kind of training first, then whether to use the settings saved in **`ml/run/default.json`** or answer the questions. Edit that one file to change the saved settings.

```bash
ml/run/train.py     # first asks which kind of training, then its settings
ml/run/watch.py     # asks what to watch
```

**Training kinds** (`ml/run/train.py` asks which):

1. **Neuroevolution** (`docs/03-neuroevolution.md`): a population evolves over generations.
2. **Reinforcement learning, DQN** (`docs/05-reinforcement-learning.md`): one network learns while it plays, about 30 seconds for 100 rounds. Watch `best.npz`, because DQN swings from round to round.

`docs/06-neuroevolution-vs-rl.md` compares the two.

**Watching** (`ml/run/watch.py` asks which):

- **single**: one player (evolution, DQN or the perfect bot); every saved generation or round is listed with its score first,
- **single VS**: an evolution player on the left against a DQN player on the right,
- **multiple**: several generations or rounds of one kind of training side by side,
- **multiple VS**: evolution generations on the left, DQN rounds on the right, one pair per row (the default is 3 x 2: evolution 0, 50, 99 against DQN 0, 99, best).

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
| `docs/` | Game design and the ML learning guide. Start with `docs/02-ml-roadmap.md`; `docs/04-brain-and-breeding.md` explains the neural network and the breeding with diagrams, `docs/05-reinforcement-learning.md` explains DQN, and `docs/06-neuroevolution-vs-rl.md` compares the two. |
| `ml/` | The brains: neural network, perfect bot, and the two trainers (neuroevolution and DQN). |
| `ml/run/` | The commands to train and watch players (`train.py`, `watch.py`) and their saved settings, `default.json`. |
| `scripts/` | The other executable launchers: `play.py`, `run_tests.py`. |
| `tests/` | Engine tests. |
| `runs/` | Training output (weights, CSV logs). Git-ignored. |

## Roadmap

1. Game engine + playable window (done)
2. Neuroevolution: a population of tiny neural nets that evolves over rounds (done)
3. Reinforcement learning (DQN) (done)
4. An LLM as a coach / policy writer
