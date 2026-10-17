import argparse

import pytest
from _args import non_negative_int, playback_speed, positive_int


@pytest.mark.parametrize("bad", ["0", "-1", "-3", "2.5", "abc", ""])
def test_positive_int_rejects(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(bad)


def test_int_checks_accept_valid_values():
    assert positive_int("1") == 1 and positive_int("50") == 50
    assert non_negative_int("0") == 0 and non_negative_int("7") == 7
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "-20", "nan", "inf", "-inf", "abc", "1001", "1e9"])
def test_speed_rejects(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        playback_speed(bad)


@pytest.mark.parametrize("good,expected", [("0.25", 0.25), ("1", 1.0), ("8", 8.0), ("20", 20.0), ("1000", 1000.0)])
def test_speed_accepts(good, expected):
    assert playback_speed(good) == expected
