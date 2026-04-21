# File Inventory (PSO + Figures)

## PSO experiment scripts (`src/experiments/watase`)

- `baseline` (20 files): base experiments and incremental historical versions
- `comparison` (6 files): label/PSO comparison and immediate-vs-delayed reference variants
- `parallel` (5 files): multiprocessing/CUDA/single-core and parallel logging variants
- `tuning` (10 files): feasibility/restart/optuna/jaccard/stochastic and constraint-focused variants
- `analysis` (4 files): analysis scripts and generation-log oriented experiment
- `debug` (2 files): debug/fix trial scripts

## Root figure assets moved to `assets/figures/root`

- `svg` (22 files): editable vector figures such as `fig1_*` and restart strategy diagrams
- `png` (46 files): rendered outputs and benchmark/result charts
- `pdf` (3 files): exported static figures

## Compatibility files kept at repository root

- `watase_24.py` -> wrapper to `src/experiments/watase/baseline/pso_4criteria_baseline.py`
- `watase_25.py` -> wrapper to `src/experiments/watase/comparison/pso_immediate_vs_delayed_reference.py`
- `watase_26.py` -> wrapper to `src/experiments/watase/analysis/pso_generation_logging.py`
- `GA_simulation_20241216.py` -> wrapper to `src/experiments/ga/GA_simulation_20241216.py`
- `GA_simulation_20241217.py` -> wrapper to `src/experiments/ga/GA_simulation_20241217.py`
- `sim241014_lts.py` -> wrapper to `src/experiments/legacy/sim241014_lts.py`
- `analyze_penalty.py` -> wrapper to `src/analysis/analyze_penalty.py`
- `soturon_graphgazou.py` -> wrapper to `scripts/figures/soturon_graphgazou.py`
- `soturon_zu1.py` -> wrapper to `scripts/figures/soturon_zu1.py`

## Additional root cleanup

- `analysis_data.csv.rtf`, `test-template-prompt.md`, `gomi.txt` are moved to `docs/archive/`
