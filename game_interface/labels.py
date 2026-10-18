"""The text shown at the top of the watcher windows. Pure functions (no pygame) so they can be tested."""


def game_label(brain, game_no, seed, replaying=False, n_games=None):
    """Label for the single-brain window.

    A brain loaded from a saved run knows its generation/round and fitness; the perfect bot
    (and any unlabelled brain) just shows its name.
    """
    if getattr(brain, "generation", None) is None:
        return f"{brain.name} | game {game_no + 1} | seed {seed}"
    if replaying:
        return f"{brain.label(fresh=False)} | game {game_no % n_games + 1}/{n_games} | seed {seed}"
    return f"{brain.label(fresh=True)} | game {game_no + 1} | seed {seed}"


def cell_label(brain):
    """Short label for one cell of the grid (VS mode sets its own tag; the perfect bot has no score)."""
    if getattr(brain, "tag", None):
        return brain.tag
    return brain.short_label()


def title_unit(brains):
    """'generations' or 'rounds', for the grid window title."""
    return "rounds" if getattr(brains[0], "unit", "gen") == "round" else "generations"
