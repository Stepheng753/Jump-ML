from types import SimpleNamespace

import pytest

from game_interface.grid import by_generation, layout, resolve_size
from ml.brains import load_generation, saved_generations


def test_default_size_is_one_row():
    assert resolve_size(4) == (1, 4)
    assert resolve_size(1) == (1, 1)


def test_size_must_match_number_of_generations():
    assert resolve_size(4, (2, 2)) == (2, 2)
    assert resolve_size(4, (4, 1)) == (4, 1)
    for bad in ((2, 3), (1, 3), (0, 4), (-2, -2)):
        with pytest.raises(ValueError):
            resolve_size(4, bad)


def test_by_generation_puts_earliest_first_whatever_the_input_order():
    b = [SimpleNamespace(generation=g, fitness=f) for g, f in [(95, 16.0), (0, 3.0), (50, 90.0), (25, 5.0)]]
    assert [x.generation for x in by_generation(b)] == [0, 25, 50, 95]


def test_layout_fits_the_window_and_has_one_origin_per_cell():
    for rows, cols in [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3)]:
        lay = layout(rows, cols)
        assert len(lay.origins) == rows * cols
        assert lay.win_w <= 1400 and lay.win_h <= 800
        xs = [x for x, _ in lay.origins]
        assert xs[:cols] == sorted(xs[:cols])                  # left to right within a row
        assert lay.origins[0][1] <= lay.origins[-1][1]         # rows go downward
    assert layout(1, 1).cell_w == 800                          # a single cell is full size


def test_load_generation_reports_which_generations_exist(tmp_path):
    import numpy as np

    from ml.brains import save_generation
    from ml.network import layer_sizes, random_genome

    rng = np.random.default_rng(0)
    sizes = layer_sizes()
    for g in (0, 5):
        save_generation(tmp_path / f"gen_{g:04d}.npz", [random_genome(rng, sizes)], [float(g)], sizes, g, [1, 2, 3])
    assert saved_generations(str(tmp_path)) == [0, 5]
    assert load_generation(str(tmp_path), 5).generation == 5
    with pytest.raises(FileNotFoundError, match="Saved generations: 0, 5"):
        load_generation(str(tmp_path), 1)
