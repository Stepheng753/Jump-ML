"""Layout helpers for watching several generations at once.

Pure functions with no pygame import, so they are easy to test and the
argument checks can run before any window opens.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

WORLD_W, WORLD_H = 800, 400   # the size one game is drawn at (see renderer.py)
GAP = 6                       # pixels between cells
BAR_H = 30                    # status bar under the grid


def resolve_size(n_cells: int, size: Optional[Sequence[int]] = None) -> Tuple[int, int]:
    """(rows, cols) for n_cells games. No --size means one row: 1 x n."""
    if size is None:
        return 1, n_cells
    rows, cols = size
    if rows < 1 or cols < 1:
        raise ValueError("--size needs two numbers that are at least 1, like --size 2 2")
    if rows * cols != n_cells:
        raise ValueError(f"--size {rows} {cols} makes {rows * cols} cells, "
                         f"but --multiple lists {n_cells} generations: they must match")
    return rows, cols


def by_generation(brains: Sequence) -> List:
    """Order brains by generation, earliest first, whatever order they were asked for in."""
    return sorted(brains, key=lambda b: b.generation)


@dataclass
class Layout:
    cell_w: int
    cell_h: int
    win_w: int
    win_h: int
    origins: List[Tuple[int, int]]   # top-left corner of each cell, row by row


def layout(rows: int, cols: int, max_w: int = 1400, max_h: int = 800) -> Layout:
    """Shrink the 800x400 game to fit a rows x cols grid inside max_w x max_h."""
    scale = min(1.0,
                (max_w - GAP * (cols + 1)) / (cols * WORLD_W),
                (max_h - BAR_H - GAP * (rows + 1)) / (rows * WORLD_H))
    cw, ch = int(WORLD_W * scale), int(WORLD_H * scale)
    origins = [(GAP + c * (cw + GAP), GAP + r * (ch + GAP)) for r in range(rows) for c in range(cols)]
    return Layout(cw, ch, cols * cw + GAP * (cols + 1), rows * ch + GAP * (rows + 1) + BAR_H, origins)
