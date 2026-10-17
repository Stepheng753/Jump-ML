"""Watch brains (perfect bot or trained nets) play in the real window.

A brain only gets the same 10 observation numbers it gets during training and
returns an action, exactly like a key press would. Nothing here is special-cased
for AI: it calls the same Game.step() the human version does.
"""
import random

import pygame

from game_engine import Game
from game_engine import config as C

from .grid import BAR_H, layout
from .renderer import AIR_OBS, BEST, GROUND_OBS, HEIGHT, SKY, TEXT, Renderer
from .sound import Sounds

SPEED_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)   # what [ and ] cycle through
RESTART_DELAY = 1.5                                   # real seconds of "game over" between games
LEADER = (90, 200, 140)


def _slower(speed):
    lower = [s for s in SPEED_STEPS if s < speed]
    return max(lower) if lower else speed


def _faster(speed):
    higher = [s for s in SPEED_STEPS if s > speed]
    return min(higher) if higher else speed


def _frame_plan(speed):
    """(game steps per drawn frame, frames per second). Above 1x we run several
    steps per frame so any speed works; below 1x we just slow the frame rate."""
    return (max(1, round(speed)), C.FPS) if speed >= 1 else (1, C.FPS * speed)


def _fresh_seed():
    """A random starting seed far above the ones training uses (run seed * 1,000,000 + ...)."""
    return random.randrange(10**8, 10**9)


def watch(brain, seed=None, speed: float = 1.0, replay: bool = False):
    """One brain, full-size window.

    Plays fresh games by default (a random seed, or --seed). With replay=True it
    plays the exact games the individual was scored on in training instead.
    """
    pygame.init()
    screen = pygame.display.set_mode((C.WORLD_W, HEIGHT))
    pygame.display.set_caption(f"Jump-ML: {brain.name}")
    clock = pygame.time.Clock()
    renderer, sounds = Renderer(screen), Sounds()

    saved_seeds = brain.seeds if replay else None
    base_seed = seed if seed is not None else _fresh_seed()

    def seed_of(n):
        return saved_seeds[n % len(saved_seeds)] if saved_seeds else base_seed + n

    def label(n):
        if saved_seeds:   # the games it was scored on, so the fitness matches what you see
            return (f"gen {brain.generation} | individual #{brain.rank} | fitness {brain.fitness:.1f}"
                    f" | game {n % len(saved_seeds) + 1}/{len(saved_seeds)} | seed {seed_of(n)}")
        who = (f"gen {brain.generation} | individual #{brain.rank} | train fit {brain.fitness:.1f}"
               if brain.generation is not None else brain.name)
        return f"{who} | game {n + 1} | seed {seed_of(n)}"

    game_no = 0
    game = Game(seed_of(game_no))
    brain.reset()
    paused, step_once, skip, over_time, best, action, dt = False, False, False, 0.0, 0, 0, 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n:
                    step_once = True
                elif event.key == pygame.K_LEFTBRACKET:
                    speed = _slower(speed)
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed = _faster(speed)
                elif event.key == pygame.K_SPACE:
                    skip = True

        steps, fps = _frame_plan(speed)
        if skip or (game.game_over and over_time >= RESTART_DELAY):
            game_no += 1
            game = Game(seed_of(game_no))
            brain.reset()
            over_time, skip = 0.0, False
        elif game.game_over:
            over_time += dt
        elif not paused or step_once:
            started = None
            for _ in range(1 if step_once else steps):
                action = brain.act(game.get_observation())
                _, _, _, info = game.step(action)
                started = info["started"] or started
                best = max(best, game.score)
                if game.game_over:
                    break
            sounds.play(started)
        step_once = False

        renderer.draw(game, best, controls=False,
                      hint=f"next game in {max(0.0, RESTART_DELAY - over_time):.1f}s")
        renderer.draw_status(f"{label(game_no)}  |  {speed:g}x{'  |  PAUSED' if paused else ''}")
        renderer.draw_brain_panel(brain.scores, action)
        pygame.display.flip()
        dt = clock.tick(fps) / 1000.0

    pygame.quit()


def watch_grid(brains, rows: int, cols: int, seed=None, speed: float = 1.0):
    """Several brains side by side, all playing the SAME obstacles.

    `brains` must already be in the order to show them: cell 1 (top-left) first,
    then left to right, top to bottom (the script passes them earliest generation
    first, so the latest ends up bottom-right). Because every
    cell plays the same seed, any difference in how they play is the brain, not
    luck. A round ends 1.5 s after every cell has crashed; there is no time cap
    here, so a brain that never crashes keeps going until you press SPACE.
    The cell with the highest live score gets a green border.
    """
    pygame.init()
    lay = layout(rows, cols)
    screen = pygame.display.set_mode((lay.win_w, lay.win_h))
    gens = ", ".join(str(b.generation) for b in brains)
    pygame.display.set_caption(f"Jump-ML: generations {gens}")
    clock = pygame.time.Clock()

    world = pygame.Surface((C.WORLD_W, HEIGHT))
    world_renderer = Renderer(world)
    scale = lay.cell_w / C.WORLD_W
    label_font = pygame.font.SysFont(None, max(16, round(30 * scale)))
    score_font = pygame.font.SysFont(None, max(22, round(46 * scale)))
    bar_font = pygame.font.SysFont(None, 24)
    crashed = pygame.Surface((lay.cell_w, lay.cell_h), pygame.SRCALPHA)
    crashed.fill((40, 14, 24, 150))

    base_seed = seed if seed is not None else _fresh_seed()
    round_no = 0
    games = [Game(base_seed) for _ in brains]
    for b in brains:
        b.reset()
    paused, step_once, skip, over_time, dt = False, False, False, 0.0, 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n:
                    step_once = True
                elif event.key == pygame.K_LEFTBRACKET:
                    speed = _slower(speed)
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed = _faster(speed)
                elif event.key == pygame.K_SPACE:
                    skip = True

        steps, fps = _frame_plan(speed)
        all_over = all(g.game_over for g in games)
        if skip or (all_over and over_time >= RESTART_DELAY):
            round_no += 1
            games = [Game(base_seed + round_no) for _ in brains]
            for b in brains:
                b.reset()
            over_time, skip = 0.0, False
        elif all_over:
            over_time += dt
        elif not paused or step_once:
            for brain, game in zip(brains, games):
                for _ in range(1 if step_once else steps):
                    if game.game_over:
                        break
                    game.step(brain.act(game.get_observation()))
        step_once = False

        screen.fill((10, 12, 20))
        top = max(g.score for g in games)
        for (x, y), brain, game in zip(lay.origins, brains, games):
            world_renderer.draw_world(game)
            screen.blit(pygame.transform.smoothscale(world, (lay.cell_w, lay.cell_h)), (x, y))
            if game.game_over:
                screen.blit(crashed, (x, y))
                msg = score_font.render("crashed", True, TEXT)
                screen.blit(msg, msg.get_rect(center=(x + lay.cell_w // 2, y + lay.cell_h // 2 - 8)))
            tag = label_font.render(f"gen {brain.generation} | #{brain.rank} | fit {brain.fitness:.1f}", True, TEXT)
            screen.blit(tag, (x + 8, y + 6))
            pts = score_font.render(str(game.score), True, TEXT)
            screen.blit(pts, (x + lay.cell_w - pts.get_width() - 10, y + 4))
            if top > 0 and game.score == top and len(games) > 1:
                pygame.draw.rect(screen, LEADER, (x, y, lay.cell_w, lay.cell_h), 3)

        status = (f"round {round_no + 1}  |  seed {base_seed + round_no}  |  {speed:g}x"
                  f"{'  |  PAUSED' if paused else ''}  |  P pause   [ ] speed   SPACE next round   Esc quit")
        bar = bar_font.render(status, True, BEST)
        screen.blit(bar, (10, lay.win_h - BAR_H + 6))
        pygame.display.flip()
        dt = clock.tick(fps) / 1000.0

    pygame.quit()
