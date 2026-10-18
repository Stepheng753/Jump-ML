#!/usr/bin/env python3
"""Run the whole test suite (pytest). It takes no arguments: just run it."""
import argparse
import sys

import _venv

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

_venv.ensure(__file__)
import pytest  # noqa: E402

sys.exit(pytest.main(["-q", _venv.ROOT + "/tests"]))
