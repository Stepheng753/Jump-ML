from typing import Optional

from . import config as C


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def points_for(speed: float, gap_prev: Optional[float], gap_next: Optional[float]) -> int:
    """1..5 points: faster and closer-to-a-neighbour obstacles are worth more.

    'Closeness' uses the smaller arrival-time gap to either neighbour; with no
    neighbour at all it counts as the loosest possible spacing.
    """
    speed_n = _clamp01((speed - C.SPEED_MIN) / (C.SPEED_MAX - C.SPEED_MIN))

    gaps = [g for g in (gap_prev, gap_next) if g is not None]
    gap = min(gaps) if gaps else C.MAX_GAP
    close_n = 1.0 - _clamp01((gap - C.MIN_GAP) / (C.MAX_GAP - C.MIN_GAP))

    score01 = C.SPEED_WEIGHT * speed_n + (1.0 - C.SPEED_WEIGHT) * close_n
    span = C.MAX_POINTS - C.MIN_POINTS
    return C.MIN_POINTS + int(round(score01 * span))


def penalty_for(distance_n: Optional[float], speed_n: Optional[float]) -> int:
    """1..5 points lost for a jump/duck that nothing needed.

    Judged by the nearest incoming obstacle at the moment of the press:
    distance_n (0 = at the player, 1 = just spawned) and speed_n (0 = slowest,
    1 = fastest). A close, fast obstacle is a partial excuse (a real threat was
    there, the press was just wrong); a far, slow one is no excuse. With no
    obstacle at all (None) there is no excuse either: the full penalty.
    """
    if distance_n is None or speed_n is None:
        excuse = 0.0
    else:
        excuse = C.PENALTY_SPEED_WEIGHT * _clamp01(speed_n) + (1.0 - C.PENALTY_SPEED_WEIGHT) * (1.0 - _clamp01(distance_n))
    span = C.MAX_PENALTY - C.MIN_PENALTY
    return C.MAX_PENALTY - int(round(excuse * span))
