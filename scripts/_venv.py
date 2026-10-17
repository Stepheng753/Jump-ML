"""Shared by the launcher scripts: re-run under the project's .venv and make the repo importable."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def ensure(script_file):
    if os.path.exists(VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(os.path.join(ROOT, ".venv")):
        os.execv(VENV_PY, [VENV_PY, os.path.abspath(script_file), *sys.argv[1:]])
    sys.path.insert(0, ROOT)
