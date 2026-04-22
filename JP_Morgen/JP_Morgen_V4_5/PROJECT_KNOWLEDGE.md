# JP Morgan Quant V4.5 Project Knowledge

Last updated: 2026-04-21

## Use This File

- This is the compact project-knowledge file for mixed Codex + Claude Code usage.
- Use it as the shared auto-loaded context file.
- For deeper context:
  - `README_PROGRESS.md` = L1 project state
  - `MODULES.md` = L2 module contracts
  - code = L3 implementation
  - `README_UPDATE.md` = historical reasoning only when needed

## Current Project State

- Active branch of research:
  - Step 2 ETF residual strategy
- Active entrypoint:
  - `strategies/step2_factor_residual_etf/main.py`
- Active trainer implementation:
  - `strategies/step1_sector_neutral/trainer.py`
- Shared model:
  - `model.py`
- Shared evaluation stack:
  - `backtest.py`
  - `visualization.py`
  - `evaluate_saved_predictions.py`

Current research conclusion:

- `4.277M` is the leading tested model size, but not yet a stable default.
- The active blocker is stability:
  - repeatability
  - seed sensitivity
  - signal-direction flips
  - regularization / validation discipline
- Keep evaluating every saved Step 2 `predictions.csv` with the same offline sweep.

## Critical Invariants

- `actual` means transformed training target.
- `raw_actual` means raw stock forward return.
- IC and ranking analysis use `actual`.
- Economic backtest and monetization use `raw_actual`.
- Step 1 and Step 2 do not use baseline `trainer.py`.
- Baseline `trainer.py` and `main.py` are benchmark reference only.

## Default Working Rules

- Split sessions by task type, not by elapsed time.
- Start each task by declaring writable files.
- Treat all non-listed files as read-only reference.
- Prefer diff-first context:
  - start from the relevant function, schema, log, or artifact sample
  - only expand to full files if needed
- Do not mix:
  - result analysis
  - architecture change
  - data-pipeline debugging
  in the same conversation unless they block each other directly.

## Task Routing

- Architecture / model change:
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `model.py`
  - active trainer
  - relevant strategy `main.py`
- Evaluation / backtest bug:
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `backtest.py`
  - `visualization.py`
  - one `predictions.csv` sample row if needed
- Data pipeline issue:
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `data_loader.py`
  - `features.py`
  - relevant `dataset.py`
- Result analysis:
  - `README_PROGRESS.md`
  - run log
  - metrics CSV
  - usually no code
- Performance work:
  - `README_PROGRESS.md`
  - `MODULES.md`
  - relevant trainer fragment
  - profile output

## Documentation Update Rules

- If project workflow or task-routing rules change, update this file.
- If module contracts or schemas change, update `MODULES.md`.
- If the active recommendation changes, update `README_PROGRESS.md`.
- If the historical log changes, append to `README_UPDATE.md`.
