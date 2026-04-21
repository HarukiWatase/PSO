# PSO Repository Handover Guide

このリポジトリは、制約付き経路探索に対する PSO（Particle Swarm Optimization）実験と比較評価、図版生成、結果分析をまとめたものです。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

代表的な実行起点:

- Baseline: `python watase_24.py`
- Immediate vs Delayed comparison: `python watase_25.py`
- Generation log variant: `python watase_26.py`

上記3つは互換ラッパーで、実体は `src/experiments/watase/` 以下にあります。

## Directory Structure

- `src/experiments/watase/baseline`: 基本実験系（過去バージョン含む）
- `src/experiments/watase/comparison`: 比較実験（label比較、参照方式比較）
- `src/experiments/watase/parallel`: 並列化/CUDA/単一コア比較
- `src/experiments/watase/tuning`: 制約・探索パラメータ・再起動・Optunaなど
- `src/experiments/watase/analysis`: 解析系とログ派生実験
- `src/experiments/watase/debug`: デバッグ/修正試行
- `src/experiments/ga`: GA系の実験スクリプト
- `src/experiments/legacy`: 過去実験の互換移行先
- `src/analysis`: 解析補助スクリプト
- `scripts/figures`: 図版生成スクリプト
- `assets/figures/root`: ルート直下にあった画像・図版の退避先（`svg`/`png`/`pdf`）
- `docs/handover`: 引き継ぎ向け資料（棚卸し、移行マップ）
- `docs/archive`: 古いメモ・一時文書の退避先
- `teruki/`: 統合アプリ側のドキュメント・依存設定

## Renaming and Compatibility

リネーム済みの中核ファイル:

- `watase_24.py` -> `src/experiments/watase/baseline/pso_4criteria_baseline.py`
- `watase_25.py` -> `src/experiments/watase/comparison/pso_immediate_vs_delayed_reference.py`
- `watase_26.py` -> `src/experiments/watase/analysis/pso_generation_logging.py`

互換性維持のため、旧名エントリポイントはルートにラッパーとして残しています（`watase_24/25/26.py`、`GA_simulation_*.py`、`sim241014_lts.py`、`analyze_penalty.py`、`soturon_*.py`）。

## Repro/Output Notes

- 実験結果は主に `Result/` 配下へ保存されます。
- 図版ファイルは新規作成時に `assets/figures/...` へ保存してください。
- 移行詳細は `docs/handover/migration_map.md` を参照してください。
- ファイル棚卸しは `docs/handover/file_inventory.md` を参照してください。

## Dependency Policy

- PSO実験の基準依存はルート `requirements.txt` を正本とします。

## Handover Operating Rules

- 新しい実験コードは目的別に `src/experiments/watase/<category>/` へ配置
- 命名は `pso_<objective>_<variant>.py` を基本形にする
- 結果追加時は `Result/` に実験条件・実行日・要約 README を残す
- 図版追加時は `assets/figures/<topic>/` に保存し、生成元スクリプトを明記する
