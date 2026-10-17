from dataclasses import dataclass

from game_engine import Game

DEFAULT_MAX_STEPS = 3600   # 1 game-minute; stops great bots from playing forever


@dataclass
class GameResult:
    score: int
    dodged: int
    steps: int
    died: bool


def play_game(brain, seed: int, max_steps: int = DEFAULT_MAX_STEPS) -> GameResult:
    """One headless game: the loop from docs/02, with no window."""
    game = Game(seed)
    brain.reset()
    obs = game.get_observation()
    steps = 0
    while steps < max_steps and not game.game_over:
        obs, _, _, _ = game.step(brain.act(obs))
        steps += 1
    return GameResult(game.score, game.dodged, steps, game.game_over)


def mean_score(brain, seeds, max_steps: int = DEFAULT_MAX_STEPS) -> float:
    return sum(play_game(brain, s, max_steps).score for s in seeds) / len(seeds)
