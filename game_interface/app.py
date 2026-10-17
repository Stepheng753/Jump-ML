"""Play the game yourself:  scripts/play.py

W / Up = jump over ground obstacles, S / Down = duck under air obstacles.
"""
import pygame

from game_engine import Action, Game
from game_engine import config as C

from .renderer import HEIGHT, Renderer
from .sound import Sounds

JUMP_KEYS = (pygame.K_w, pygame.K_UP)
DUCK_KEYS = (pygame.K_s, pygame.K_DOWN)


def main():
    pygame.init()
    screen = pygame.display.set_mode((C.WORLD_W, HEIGHT))
    pygame.display.set_caption("Jump-ML")
    clock = pygame.time.Clock()
    renderer, sounds, game = Renderer(screen), Sounds(), Game()
    high_score = 0

    running = True
    while running:
        action = Action.NONE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in JUMP_KEYS:
                    action = Action.JUMP
                elif event.key in DUCK_KEYS:
                    action = Action.DUCK
                elif event.key == pygame.K_SPACE and game.game_over:
                    game.reset()

        _, _, _, info = game.step(action)
        sounds.play(info["started"])
        high_score = max(high_score, game.score)

        renderer.draw(game, high_score)
        pygame.display.flip()
        clock.tick(C.FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
