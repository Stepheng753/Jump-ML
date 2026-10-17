#!/usr/bin/env python3
"""Run the test suite (pytest). Any extra arguments are passed to pytest:

    scripts/run_tests.py
    scripts/run_tests.py -v
    scripts/run_tests.py -k perfect
"""
import argparse
import sys

import _venv

_, pytest_args = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
).parse_known_args()

_venv.ensure(__file__)
import pytest  # noqa: E402

sys.exit(pytest.main(["-q", _venv.ROOT + "/tests", *pytest_args]))
