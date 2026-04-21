"""Legacy entrypoint for the renamed PSO generation-log script."""

from pathlib import Path
import runpy


_TARGET = Path(__file__).resolve().parent / "src/experiments/watase/analysis/pso_generation_logging.py"


if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    globals().update(runpy.run_path(str(_TARGET)))
