# game_engine

The rules of the game and nothing else: **no pygame, no sound, no real-time clock**. Because it never draws anything it runs about 200,000 steps per second, which is what makes training thousands of games practical. A test (`test_engine_has_no_pygame`) fails if anything in here imports pygame.

## Using it

```python
from game_engine import Game, Action

game = Game(seed=1)                       # same seed = same obstacles, every time
obs = game.reset()                        # 10 numbers the AI can see
obs, reward, done, info = game.step(Action.JUMP)   # advance exactly one frame (1/60 s)
```

`step(action)` returns:

| Value | Meaning |
| --- | --- |
| `obs` | the new observation (10 numbers, see `docs/04-brain-and-breeding.md`) |
| `reward` | points earned this frame, minus any wasted-action penalty, minus 10 on the frame you crash |
| `done` | `True` once you have crashed |
| `info` | `info["started"]` is `"jump"`/`"duck"` on the frame an action begins (the window uses it for sound); `info["points"]` is points earned this frame; `info["penalty"]` is points lost this frame to a jump/duck nothing needed |

`Action` is `NONE = 0`, `JUMP = 1`, `DUCK = 2`. Pressing during a jump or duck is ignored until it ends.

## Files

| File | What it holds |
| --- | --- |
| `config.py` | every tunable number: speeds, gaps, jump height and time, obstacle sizes, points range. **Change the game here.** |
| `entities.py` | `Action`, `PlayerState`, `Kind` enums; `Player` (jump/duck state and hitbox); `Obstacle`; `overlaps()` for collisions |
| `spawner.py` | plans obstacles on an **arrival timeline**: each one gets a random speed and kind, and its spawn moment is worked out backwards from when it should reach the player. Arrivals are at least 0.9 s apart, so the game is always beatable and a fast obstacle never catches a slow one |
| `scoring.py` | `points_for(speed, gap_prev, gap_next)` gives 1 to 5 points: faster and closer-to-a-neighbour obstacles are worth more. `penalty_for(distance, speed)` gives the 1 to 5 points lost for a jump/duck nothing needed |
| `game.py` | `Game` (`reset`, `step`, `get_observation`) and `OBS_SIZE` |

## Rules worth knowing

- The world is 800 px wide. The player stands at x = 400; obstacles spawn at x = 800 and move **left**.
- Ground obstacles (orange) are cleared by jumping. Air obstacles (blue) hang from 40 to 120 px, higher than the 100 px jump peak, so they can only be cleared by ducking.
- A jump or duck that no obstacle needed costs 1 to 5 points (more when the nearest obstacle is far and slow). `game.wasted` counts them. The score can go negative.
- Full details and the scoring formulas are in `docs/01-game-design.md`.

## Tests

`tests/test_engine.py` covers scoring bounds, spacing, reproducibility, the one-hot observation, and a scripted perfect-timing bot that must survive 2 minutes on 30 different seeds (proof the spawner never makes an impossible pattern).
