"""Legacy entrypoint for the renamed PSO comparison script."""

from pathlib import Path
import runpy


_TARGET = Path(__file__).resolve().parent / "src/experiments/watase/comparison/pso_immediate_vs_delayed_reference.py"


if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    globals().update(runpy.run_path(str(_TARGET)))
