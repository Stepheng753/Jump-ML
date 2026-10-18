"""The answer checks the question prompts use: a bad answer raises an error that explains what is wrong."""
import argparse
import math

MAX_SPEED = 1000.0   # beyond this a "frame" would hold more game steps than the engine can run per frame


def positive_int(text):
    value = _whole_number(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {value})")
    return value


def non_negative_int(text):
    value = _whole_number(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more (got {value})")
    return value


def any_int(text):
    return _whole_number(text)


def run_name(text):
    """A run folder name: a plain name, not a path."""
    if not text or "/" in text or "\\" in text or text in (".", ".."):
        raise argparse.ArgumentTypeError("use a plain folder name, like xyz")
    return text


def positive_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a number greater than 0 (got {text})")
    return value


def gamma_value(text):
    """The DQN discount: 0 up to (not including) 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"must be at least 0 and below 1 (got {text})")
    return value


def playback_speed(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a number greater than 0 (got {text})")
    if value > MAX_SPEED:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SPEED:g} (got {text})")
    return value


def _whole_number(text):
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number")
