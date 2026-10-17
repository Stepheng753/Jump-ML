"""All tunable numbers for the game in one place.

Units: pixels for distance, seconds for time. The engine runs at a fixed
timestep so it behaves identically for a human, a bot, or a headless trainer.
"""

FPS = 60
DT = 1.0 / FPS

# World layout. Heights are measured upward from the ground (y = 0).
WORLD_W = 800
SPAWN_X = WORLD_W           # obstacles appear at the right edge...
PLAYER_X = 400              # ...and travel left toward the player
TRAVEL_DIST = SPAWN_X - PLAYER_X

# Player
PLAYER_W = 24
STAND_H = 60
DUCK_H = 30
JUMP_PEAK = 100
JUMP_TIME = 0.6             # seconds in the air
DUCK_TIME = 0.5             # seconds ducking

# Obstacles
OBSTACLE_W = 24
GROUND_OBS_H = 40           # sits on the ground -> must jump
AIR_OBS_BOTTOM = 40         # floats above the ground -> must duck (stand=60 hits, duck=30 clears)
AIR_OBS_H = 80             # top (120) is above the jump peak (100), so it cannot be jumped over
SPEED_MIN = 220.0           # px/s
SPEED_MAX = 520.0

# Spacing is measured in *arrival time*: seconds between two obstacles
# reaching the player. MIN_GAP guarantees there is time to react.
MIN_GAP = 0.9
MAX_GAP = 2.2
FIRST_ARRIVAL = 1.5         # grace period before the first obstacle arrives

# Scoring
MIN_POINTS = 1
MAX_POINTS = 5
SPEED_WEIGHT = 0.5          # how much speed vs. closeness matters (0..1)
DEATH_PENALTY = 10.0        # used by the ML reward only

# Wasted actions: a jump/duck that no obstacle needed costs points
MIN_PENALTY = 1
MAX_PENALTY = 5
PENALTY_SPEED_WEIGHT = 0.5  # speed vs. closeness of the nearest obstacle (0..1); both lower the penalty
