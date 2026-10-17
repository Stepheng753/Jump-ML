import numpy as np
import pygame

from game_engine import Game, Kind, PlayerState
from game_engine import config as C

HEIGHT = 400
GROUND_Y = 300   # screen y of the ground line

SKY = (24, 28, 44)
GROUND = (70, 76, 100)
PLAYER = (90, 200, 140)
GROUND_OBS = (230, 120, 80)
AIR_OBS = (120, 160, 240)
TEXT = (240, 240, 245)
BEST = (150, 156, 185)
PENALTY = (230, 90, 90)
PENALTY_FLASH = 1.0   # seconds the '-N' stays on screen

INSTRUCTIONS = (
    ("W / Up Arrow: jump", GROUND_OBS),
    ("S / Dn Arrow: duck", AIR_OBS),
)


def _to_screen(rect):
    x0, y0, x1, y1 = rect
    return pygame.Rect(round(x0), round(GROUND_Y - y1), round(x1 - x0), round(y1 - y0))


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 36)
        self.big = pygame.font.SysFont(None, 64)
        self.small = pygame.font.SysFont(None, 26)

    def draw_world(self, game: Game):
        """Just the playing field: sky, ground, obstacles, player. No text."""
        s = self.screen
        s.fill(SKY)
        pygame.draw.rect(s, GROUND, (0, GROUND_Y, C.WORLD_W, HEIGHT - GROUND_Y))

        for ob in game.obstacles:
            color = GROUND_OBS if ob.kind == Kind.GROUND else AIR_OBS
            pygame.draw.rect(s, color, _to_screen(ob.rect), border_radius=4)

        body = _to_screen(game.player.rect)
        pygame.draw.rect(s, PLAYER, body, border_radius=6)
        eye_y = body.y + (8 if game.player.state != PlayerState.DUCK else 5)
        pygame.draw.circle(s, SKY, (body.centerx + 5, eye_y), 3)

    def draw(self, game: Game, high_score: int, controls: bool = True, hint: str = "Press SPACE to restart"):
        s = self.screen
        self.draw_world(game)

        score = self.font.render(f"Score {game.score}", True, TEXT)
        s.blit(score, (C.WORLD_W - score.get_width() - 16, 14))
        if game.last_penalty and game.t - game.last_penalty_at < PENALTY_FLASH:
            lost = self.font.render(f"-{game.last_penalty}", True, PENALTY)
            s.blit(lost, (C.WORLD_W - score.get_width() - 16 - lost.get_width() - 12, 14))
        best = self.font.render(f"Best {high_score}", True, BEST)
        s.blit(best, (C.WORLD_W - best.get_width() - 16, 14 + score.get_height() + 6))

        if controls:
            for i, (line, color) in enumerate(INSTRUCTIONS):
                text = self.small.render(line, True, color)
                s.blit(text, (16, 14 + i * (text.get_height() + 4)))

        if game.game_over:
            msg = self.big.render("Game over", True, TEXT)
            hint = self.font.render(hint, True, TEXT)
            s.blit(msg, msg.get_rect(center=(C.WORLD_W // 2, 130)))
            s.blit(hint, hint.get_rect(center=(C.WORLD_W // 2, 180)))

    def draw_status(self, text: str):
        """Small line at the top-left: who is playing, seed, speed."""
        self.screen.blit(self.small.render(text, True, BEST), (16, 14))

    def draw_brain_panel(self, scores, action: int):
        """What the brain 'thinks' this frame, drawn on the ground strip.

        For a net: its 3 raw output scores as bars (bar length = softmax share,
        number = raw score); the winner is highlighted. For the perfect bot
        there are no scores, so just show the action it chose.
        """
        s, x, y = self.screen, 16, GROUND_Y + 14
        names = ("nothing", "jump", "duck")
        colors = (BEST, GROUND_OBS, AIR_OBS)
        if scores is None:
            label = self.small.render(f"action: {names[action]}", True, colors[action] if action else BEST)
            s.blit(label, (x, y))
            return
        share = np.exp(scores - np.max(scores))
        share /= share.sum()
        for i, name in enumerate(names):
            chosen = i == action
            row_y = y + i * 24
            text = self.small.render(f"{name:<8}{scores[i]:+.2f}", True, TEXT if chosen else BEST)
            s.blit(text, (x, row_y))
            pygame.draw.rect(s, GROUND, (x + 150, row_y + 3, 160, 14), border_radius=3)
            pygame.draw.rect(s, colors[i], (x + 150, row_y + 3, int(160 * share[i]), 14), border_radius=3)
