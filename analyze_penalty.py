"""Legacy entrypoint for migrated script."""

from pathlib import Path
import runpy

_TARGET = Path(__file__).resolve().parent / "src/analysis/analyze_penalty.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    globals().update(runpy.run_path(str(_TARGET)))
