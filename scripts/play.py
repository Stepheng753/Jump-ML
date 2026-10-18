#!/usr/bin/env python3
"""Play the game yourself. (It takes no arguments: just run it.)

Keys: W / Up = jump (orange obstacles), S / Down = duck (blue obstacles),
SPACE = restart after a crash, Esc = quit.
"""
import argparse

import _venv

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

_venv.ensure(__file__)
from game_interface.app import main  # noqa: E402

main()
