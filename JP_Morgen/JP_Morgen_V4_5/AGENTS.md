# JP Morgan Quant V4.5 Agent Guide

Scope:

- Applies to `JP_Morgen/JP_Morgen_V4_5/`
- This is the single authoritative agent document for both Codex and Claude Code.
- Do not maintain a second independent agent policy file for the same scope.

Default context stack:

1. `PROJECT_KNOWLEDGE.md`
2. `README_PROGRESS.md`
3. `MODULES.md`
4. Only the code files that will actually be edited or debugged
5. `README_UPDATE.md` only when the historical reason for a decision matters

Layer meaning:

- `PROJECT_KNOWLEDGE.md`
  - project-level working rules and fast-start context
- `README_PROGRESS.md`
  - L1 project state
- `MODULES.md`
  - L2 module contracts
- code
  - L3 implementation details
- `README_UPDATE.md`
  - chronological history, only when needed

Conversation discipline:

- Split sessions by task type, not by elapsed time.
- At the top of each task, declare writable files explicitly.
- Treat non-listed files as read-only reference unless the task truly expands.
- Prefer diff-first context:
  - start from the relevant function, schema, or artifact sample
  - do not dump whole files unless needed
- Do not mix result analysis, architecture changes, and data-pipeline debugging in one conversation unless they block each other directly.

Task templates:

- Architecture / model change:
  - `PROJECT_KNOWLEDGE.md`
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `model.py`
  - active trainer
  - relevant strategy `main.py`
- Evaluation / backtest bug:
  - `PROJECT_KNOWLEDGE.md`
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `backtest.py`
  - `visualization.py`
  - one `predictions.csv` sample row if schema is in question
- Data pipeline issue:
  - `PROJECT_KNOWLEDGE.md`
  - `README_PROGRESS.md`
  - `MODULES.md`
  - `data_loader.py`
  - `features.py`
  - relevant `dataset.py`
- Result analysis:
  - `PROJECT_KNOWLEDGE.md`
  - `README_PROGRESS.md`
  - run log
  - metrics CSV
  - no code by default
- Performance work:
  - `PROJECT_KNOWLEDGE.md`
  - `README_PROGRESS.md`
  - `MODULES.md`
  - relevant trainer fragment
  - profile output

Project-specific invariants:

- Active research path is Step 2 ETF residual:
  - `strategies/step2_factor_residual_etf/main.py`
- Active trainer logic lives in:
  - `strategies/step1_sector_neutral/trainer.py`
- Step 2 trainer file is a thin re-export wrapper:
  - `strategies/step2_factor_residual_etf/trainer.py`
- `actual` is for IC and target-space evaluation.
- `raw_actual` is for realized-return economics and backtest.
- Baseline `trainer.py` is a legacy benchmark path, not the active optimization path.
- Current Step 2 `predictions.csv` may contain trade-signal columns, but shared offline evaluation only relies on:
  - `ticker`
  - `date`
  - `predicted`
  - `actual`
  - `raw_actual`

Documentation rules:

- If project-level working rules change, update `PROJECT_KNOWLEDGE.md`.
- If shapes, schemas, defaults, or module ownership change, update `MODULES.md` in the same change.
- If the recommended next step or project interpretation changes, update `README_PROGRESS.md`.
- If the implementation history or decision record changes, append a node to `README_UPDATE.md`.
