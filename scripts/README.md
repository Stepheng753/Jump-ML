# scripts

The general commands. Each is an executable file with a shebang, so run it directly from the project root (no `python` needed), and it works from any directory.

| Script | What it does |
| --- | --- |
| `play.py` | play the game yourself |
| `run_tests.py` | run the whole test suite |

```bash
scripts/play.py
scripts/run_tests.py
```

Neither takes arguments. (`--help` still works, but only describes the script.)

**Training and watching live in `ml/run/`**: `ml/run/train.py` and `ml/run/watch.py` ask you questions (or read the saved settings in `ml/run/default.json`). See `ml/run/README.md`.

## Helper file (not a command)

| File | Purpose |
| --- | --- |
| `_venv.py` | makes each script re-run itself inside the project's `.venv` and puts the project root on the import path, so they work from any directory without activating anything. Every script uses it, including the two in `ml/run/` |

The leading underscore marks it as internal.
