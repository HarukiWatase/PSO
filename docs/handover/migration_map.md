# Migration Map

## Renamed core scripts

- `watase_24.py` -> `src/experiments/watase/baseline/pso_4criteria_baseline.py`
- `watase_25.py` -> `src/experiments/watase/comparison/pso_immediate_vs_delayed_reference.py`
- `watase_26.py` -> `src/experiments/watase/analysis/pso_generation_logging.py`

## Categorized script relocation

- Former `watase_*.py` scripts are moved under `src/experiments/watase/` and split by role:
  - `baseline`, `comparison`, `parallel`, `tuning`, `analysis`, `debug`
- Legacy script names are preserved only for `watase_24.py`, `watase_25.py`, `watase_26.py` as root wrappers for transition.

## Figure asset relocation

- Root-level `*.svg` -> `assets/figures/root/svg/`
- Root-level `*.png` -> `assets/figures/root/png/`
- Root-level `*.pdf` -> `assets/figures/root/pdf/`

## Additional root script relocation

- `GA_simulation_20241216.py` -> `src/experiments/ga/GA_simulation_20241216.py`
- `GA_simulation_20241217.py` -> `src/experiments/ga/GA_simulation_20241217.py`
- `sim241014_lts.py` -> `src/experiments/legacy/sim241014_lts.py`
- `analyze_penalty.py` -> `src/analysis/analyze_penalty.py`
- `soturon_graphgazou.py` -> `scripts/figures/soturon_graphgazou.py`
- `soturon_zu1.py` -> `scripts/figures/soturon_zu1.py`
- `analysis_data.csv.rtf` -> `docs/archive/analysis_data.csv.rtf`
- `test-template-prompt.md` -> `docs/archive/test-template-prompt.md`
- `gomi.txt` -> `docs/archive/gomi.txt`

## Naming rules (new files)

- PSO script names should describe intent and axis, for example:
  - `pso_<objective>_<variant>.py`
  - `pso_4criteria_baseline.py`
  - `pso_immediate_vs_delayed_reference.py`
- Analysis scripts should include output semantics, for example:
  - `*_generation_logging.py`
  - `*_compare.py`
