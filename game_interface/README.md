# game_interface

Everything you can see, hear, or press. It is a thin skin over `game_engine`: it asks the engine to step and draws the result. The engine knows nothing about this folder.

You normally start it through the launchers in `scripts/` (`scripts/play.py`, `ml/run/watch.py`), not directly.

## Files

| File | What it does |
| --- | --- |
| `app.py` | the human game. Reads `W`/`↑` (jump) and `S`/`↓` (duck), steps the engine at 60 FPS, plays sounds |
| `watcher.py` | watches a **brain** play. `watch()` shows one brain full size; `watch_grid()` shows several side by side, all playing the same obstacles |
| `renderer.py` | `Renderer`: `draw_world` (field only), `draw` (field plus score and controls), `draw_status` (label line), `draw_brain_panel` (the net's 3 output scores) |
| `sound.py` | jump (rising) and duck (falling) tones generated with numpy, so no audio files. If there is no audio device it runs silently and prints one warning |
| `plan.py` | carries out the watch plan that the `ml/run/watch.py` questions produce: `build_watch` loads the players (no pygame), `launch` opens the window |
| `labels.py` | the label text for the windows (pure, no pygame): generation or round, fitness or eval score, game number, seed |
| `grid.py` | layout maths for the grid view: `resolve_size`, `by_generation`, `layout`. **Pure functions with no pygame import**, so they can be tested and validated before any window opens |

## What a brain is

The watcher works with anything that has this shape (see `ml/brains.py`):

```python
brain.name              # text for the window title
brain.reset()           # called at the start of each game
brain.act(observation)  # the 10 numbers in, an action (0, 1 or 2) out
brain.scores            # optional: 3 raw scores, drawn as bars (None for the perfect bot)
```

A brain's `act()` result goes into the same `Game.step()` your keypress does. Nothing is special-cased for AI.

## Keys in the watcher

`P` pause, `N` one frame while paused, `[` and `]` slower/faster, `SPACE` skip to the next game (next round in the grid), `Esc` quit. Speeds above 1x run several game steps per drawn frame, so a playback speed of 20 works.

## Window sizes

The single view is 800 x 400. The grid shrinks each cell to fit inside 1400 x 800, with a status bar underneath. The grid has no sound (several overlapping jump noises would just be noise).
