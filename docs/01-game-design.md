# Game design

All numbers live in `game_engine/config.py`; change them there.

## The world
- 800 px wide. The player is fixed at x = 400. Obstacles spawn at the right edge (x = 800) and move **left** toward the player.
- Time advances in fixed 1/60 s steps (`Game.step(action)`), so a human, a bot, and a headless trainer all see identical physics.

## Player
| Action | Key | Effect |
| --- | --- | --- |
| Jump | `W` / `↑` | Parabolic hop, 0.6 s, peak 100 px |
| Duck | `S` / `↓` | Hitbox shrinks from 60 to 30 px for 0.5 s |

While a jump or duck is in progress, new presses are ignored. The player must land first.

## Obstacles
- **Ground** (40 px tall, on the floor): the player must be in the air. Ducking or standing hits it.
- **Air** (floating 40–120 px up): the player must be ducking. Standing hits it, and jumping hits it too, because its top is above the 100 px jump peak.
- Speed is random per obstacle, 220–520 px/s.

## Spawning
The spawner plans obstacles by **arrival time**, meaning when each one reaches the player. Each obstacle's arrival is 0.9–2.2 s after the previous one, and its spawn moment is worked out backwards from its speed. Consequences:
- A fast obstacle can never catch up to a slow one in front of it.
- There is always time to finish one action before the next obstacle arrives.
- `tests/test_engine.py` proves the second point: a scripted bot with perfect timing survives 2 minutes on 30 different seeds.

## Scoring
Each dodged obstacle gives 1–5 points:

```
speed_n = (speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)        # 0 slow .. 1 fast
close_n = 1 - (smallest_gap_to_neighbour - MIN_GAP) / (MAX_GAP - MIN_GAP)   # 1 = very close
points  = 1 + round(4 * (0.5 * speed_n + 0.5 * close_n))
```

`SPEED_WEIGHT` in the config changes the 50/50 blend. See `game_engine/scoring.py`.

## Wasted jumps and ducks

A jump or duck that nothing needed **costs 1 to 5 points**, taken the moment you press. This stops a player from just jumping or ducking all the time.

**"Needed"** means that, while the action lasts (0.6 s for a jump, 0.5 s for a duck), an obstacle of the matching kind (ground for a jump, air for a duck) will pass through the player's column. Obstacles move at a constant speed, so the game works this out on the spot. A jump at an air obstacle, a duck at a ground obstacle, or a jump with the next obstacle still too far away to arrive before it lands, are all wasted. A press made while already mid-jump or mid-duck is ignored and costs nothing.

**How much** depends on the nearest incoming obstacle at the moment you pressed, using its distance and speed:

```
close_n = 1 - distance / travel_distance     # 1 = right at you, 0 = just spawned
excuse  = 0.5 * speed_n + 0.5 * close_n      # 1 = close and fast, 0 = far and slow (or no obstacle at all)
penalty = 5 - round(4 * excuse)              # 5 = no excuse ... 1 = nearly justified
```

So a press with nothing around, or with a far, slow obstacle, costs the full 5; a press with a close, fast obstacle costs as little as 1, because a real threat was there. The score can go below zero. The window shows the loss as a red `-N` beside the score for a second. `MIN_PENALTY`, `MAX_PENALTY` and `PENALTY_SPEED_WEIGHT` in `game_engine/config.py` tune it, and the formula is `penalty_for` in `game_engine/scoring.py`.

## What an AI sees
`Game.get_observation()` returns 10 numbers, roughly 0..1: the player's state as three 0/1 flags (running, jumping, ducking), then the jump/duck progress, then for the 2 nearest incoming obstacles their distance, speed and type (ground/air). That is all an AI gets, with no pixels.
