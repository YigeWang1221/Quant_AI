# JP Morgan Quant V4.5 Module Contracts

Last updated: 2026-04-21

## Purpose

- This file is the L2 context layer for the project.
- Pair it with `README_PROGRESS.md` in every new AI conversation.
- Use `README_UPDATE.md` only when the historical reason for a decision matters.
- Update this file whenever shapes, schemas, defaults, artifact meaning, or active module ownership change.

## Active Path

Current research path:

- Entrypoint:
  - `strategies/step2_factor_residual_etf/main.py`
- Dataset:
  - `strategies/step2_factor_residual_etf/dataset.py`
- Active trainer implementation:
  - `strategies/step1_sector_neutral/trainer.py`
- Step 2 trainer wrapper:
  - `strategies/step2_factor_residual_etf/trainer.py`
  - re-exports the Step 1 trainer
- Shared model / evaluation:
  - `model.py`
  - `backtest.py`
  - `visualization.py`
  - `evaluate_saved_predictions.py`
- Active remote launcher:
  - `strategies/step2_factor_residual_etf/run.sh`

Project invariants:

- IC logic uses `actual`.
- Economic backtest logic uses `raw_actual` when present.
- Step 1 and Step 2 do not use baseline `trainer.py`.
- Active trainer = Node 09 batched-day path + Node 20 stability controls.
- Local code default `batch_days = 20`; Step 2 `scale_4_277M` and `stability_*` presets override it to `16` in `run.sh`.

## Shared Contracts

### Raw inputs

`data_loader.get_or_load_data(start, end) -> (raw_data, market_data, fundamentals)`

- `raw_data`
  - `dict[str, pandas.DataFrame]`
  - key = stock ticker
  - index = `DatetimeIndex`
  - expected numeric columns = `Open`, `High`, `Low`, `Close`, `Volume`
- `market_data`
  - close arrays plus matching `*_idx` date arrays
  - current reference keys:
    - `SP500`
    - `Nasdaq100`
    - `EnergySector`
    - `Bond20Y`
    - `Gold`
- `fundamentals`
  - per-ticker scalar dict
  - current keys:
    - `pe`
    - `pb`
    - `market_cap`
    - `dividend_yield`
    - `roe`

### Factor frame

`features.compute_v3_factors(df, market_data, fund_data) -> DataFrame(T, F)`

- Output index stays on stock dates after factor warmup.
- Current code usually yields `35` factors when all three relative-market references exist.
- `F` is still dynamic:
  - downstream code trusts `full_data["num_factors"]`
- Output is forward-filled, then residual missing values are filled with `0.0`.

### Normalized training data

All dataset builders emit `full_data` with this shared core contract:

```text
full_data = {
  "daily_samples": {date -> sample},
  "valid_dates": [date1, date2, ...],
  "tickers_list": [ticker1, ticker2, ...],
  "num_factors": F,
  ...strategy-specific metadata...
}

sample = {
  "X": float32 array (N, 20, F),
  "y": float32 array (N,),
  "raw_y": float32 array (N,),
  "tickers": list[str] length N,
}
```

Shared dataset filters:

- lookback = `20`
- rolling scaler window = `80`
- minimum history = `4` years
- valid date requires at least `80%` of surviving stocks present
- trainable date requires at least `20` names

### Fold contract

`generate_folds(...) -> list[fold]`

Each `fold` contains:

- `train_start`
- `train_end`
- `val_start`
- `val_end`
- `test_start`
- `test_end`
- `n_train`
- `n_val`
- `n_test`

Date semantics in `get_dates_in_range()`:

- start inclusive
- end exclusive

### Output schemas

Minimum `predictions.csv` schema trusted across tools:

- `ticker`
- `date`
- `predicted`
- `actual`
- `raw_actual`

Current Step 2 `predictions.csv` after Node 14 also adds:

- `signal_raw`
- `trade_signal_raw`
- `trade_signal`
- `trade_signal_active`

Column meaning:

- `predicted`
  - raw model score
- `actual`
  - transformed training label
  - use for IC and target-space analysis
- `raw_actual`
  - raw stock forward return
  - use for economics and backtest
- `trade_signal_active`
  - signal actually traded after original-vs-reversed selection

Active Step 1 / Step 2 `fold_metrics.csv` columns:

- `fold`
- `test_start`
- `test_end`
- `val_ic`
- `test_ic`
- `fold_seed`
- `best_epoch`
- `n_samples`

`backtest.csv` columns:

- `date`
- `long_return`
- `short_return`
- `gross_return`
- `net_return`
- `turnover`
- `cost`
- `return_source`
- `avg_long_signal`
- `avg_short_signal`
- `top_n`
- `rebal_freq`
- `score_smooth_window`
- `score_smooth_method`
- `no_trade_band`

## File Contracts

- `config.py`
  - global constants for universe, date range, directories, and shared defaults
  - important defaults:
    - `DEFAULT_LISTNET_WEIGHT = 0.0`
    - `DEFAULT_BATCH_DAYS = 20`
    - `DEFAULT_AMP_MODE = "on"`
  - last major interface change for active path:
    - Node 09

- `data_loader.py`
  - cache-aware loader / downloader for `v3_data/`
  - falls back to `yfinance` only when cache is missing or requested range is uncovered
  - changing cache layout or reference series format impacts every dataset builder

- `features.py`
  - shared factor engineering for baseline, Step 1, and Step 2
  - any feature-column change changes `num_factors` for all downstream models

- `dataset.py`
  - baseline dataset builder
  - target = raw 5-day forward return
  - `raw_target == target`
  - contains `DailyDataset`, but the active Step 1 / Step 2 trainer path does not rely on it
  - treat as legacy benchmark support

- `strategies/step1_sector_neutral/dataset.py`
  - Step 1 label = stock raw 5-day forward return minus same-day sector mean forward return
  - missing sector-map names fall back to raw target
  - adds `target_strategy = "sector_neutral"`
  - stable secondary strategy branch

- `strategies/step2_factor_residual_etf/dataset.py`
  - current active data path
  - label = stock raw 5-day forward return minus ETF-implied 5-day return using trailing daily-return betas
  - ETF proxies:
    - `SPY`
    - `QQQ`
    - `XLE`
    - `TLT`
    - `GLD`
  - if future factor returns are missing or betas cannot be estimated, that day falls back to raw target
  - adds:
    - `target_strategy = "factor_residual_etf"`
    - `beta_window`
    - `beta_min_obs`
    - `etf_proxy_tickers`
  - invariant:
    - `actual` = ETF-residual target
    - `raw_actual` = raw stock forward return

- `model.py`
  - `QuantV4.forward(x, stock_mask=None)` accepts:
    - `x: (N, T, F)` and returns `(N,)`
    - `x: (B, N, T, F)` and returns `(B, N)`
  - `stock_mask` matches `(N,)` or `(B, N)`
  - `TwoWayBlock` does temporal attention over `T`, then cross-sectional attention over `N`
  - `stock_mask` is used only for the cross-sectional padding mask
  - AMP-safe in the current CUDA path
  - last major interface change:
    - Node 09 batched-day tensor support

- `loss.py`
  - `V4CombinedLoss.forward(pred, target, mask=None) -> (loss, listnet_loss, ic)`
  - single-date contract:
    - `pred: (N,)`
    - `target: (N,)`
  - active default `listnet_weight = 0.0`
  - implication:
    - active trainer uses vectorized IC-only logic instead of looping through `criterion()` day by day

- `strategies/step1_sector_neutral/trainer.py`
  - active trainer for both Step 1 and Step 2
  - `train_one_fold(...) -> (result_df, best_val_ic, test_ic, fold_seed, best_epoch)`
  - behavior:
    - precompute padded day tensors on CPU
    - batch `batch_days` dates into one forward pass
    - move only the current batch to device
    - batch validation and test inference by date groups
    - cast predictions back to fp32 before loss
  - fold seed rule:
    - `base_seed + fold_index - 1`
  - result frame columns:
    - `ticker`
    - `date`
    - `predicted`
    - `actual`
    - `raw_actual`
  - high-impact changes here affect both Step 1 and Step 2
  - last major changes:
    - Node 09
    - Node 20

- `strategies/step2_factor_residual_etf/trainer.py`
  - thin compatibility wrapper only
  - re-exports `describe_folds`, `generate_folds`, `train_one_fold`
  - do not let behavior drift here unless Step 2 truly needs a separate trainer

- `trainer.py`
  - baseline trainer only
  - preloads all train-day tensors onto device
  - validation and test still run day by day
  - more VRAM-heavy than the active trainer path
  - treat as legacy benchmark reference

- `backtest.py`
  - `prepare_backtest_predictions()` adds:
    - `signal_raw`
    - `trade_signal_raw`
    - `trade_signal`
  - `backtest_from_predictions()`:
    - uses `raw_actual` for realized PnL when present
    - falls back to `actual` only for older outputs
    - auto-prepares signals if `trade_signal` is missing
    - `rev=True` flips direction after signal preparation
    - `no_trade_band` gives incumbency bonus during replacement
  - last major behavior changes:
    - Node 05
    - Node 14

- `visualization.py`
  - `plot_fold_ic()` and `plot_daily_ic()` use `predicted` vs `actual`
  - `plot_backtest()` uses `raw_actual` when present for scatter and quantile-return panels
  - callers should pass `predicted` already aligned to the chosen direction if reversal was selected

- `main.py`
  - baseline end-to-end entrypoint
  - uses baseline dataset + baseline trainer + original/reversed direction test
  - saves:
    - `fold_metrics.csv`
    - `predictions.csv`
    - `backtest.csv`
  - benchmark only

- `strategies/step1_sector_neutral/main.py`
  - Step 1 end-to-end entrypoint
  - uses the active shared trainer
  - accepts stability controls:
    - `--seed`
    - `--deterministic`
    - `--weight_decay`
    - `--grad_clip_norm`
    - `--early_stop_min_delta`
  - last major interface change:
    - Node 21

- `strategies/step2_factor_residual_etf/main.py`
  - current active orchestration file
  - pipeline:
    - build Step 2 dataset
    - generate folds
    - train through shared active trainer
    - aggregate fold outputs
    - build trade signals
    - compare original vs reversed direction
    - plot and save artifacts
  - Step 2-specific controls:
    - `--beta_window`
    - `--beta_min_obs`
    - `--score_smooth_window`
    - `--score_smooth_method`
    - `--no_trade_band`
  - also accepts Node 20 stability controls
  - saves:
    - `fold_metrics.csv`
    - `predictions.csv`
    - `backtest.csv`
    - `img/`
    - `run_manifest.json`
    - `run_manifest.txt`
  - direction selection rule:
    - choose original vs reversed by final cumulative `net_return`
    - persist chosen tradable signal in `trade_signal_active`
  - last major changes:
    - Node 14
    - Node 20
    - Node 21

- `evaluate_saved_predictions.py`
  - offline sweep over saved `predictions.csv`
  - accepts:
    - one run dir
    - one directory tree
    - one or more direct `predictions.csv` files
  - minimum required columns:
    - `ticker`
    - `date`
    - `predicted`
    - `actual`
  - `raw_actual` is strongly preferred for economic validity
  - ignores extra columns beyond the base five fields
  - sweeps:
    - `top_n`
    - `rebal_freq`
    - `score_smooth_method`
    - `score_smooth_window`
    - `no_trade_band`
  - `direction_mode="auto"` keeps the better direction by `net_total`
  - outputs:
    - `sweep_summary.csv`
    - `best_config_by_run.csv`
    - `best_config_yearly.csv`
    - `sweep_manifest.json`
  - added in Node 17

- `utils.py`
  - runtime helpers for logging, device selection, artifact dirs, and seeding
  - `set_random_seed(seed, deterministic=False)` syncs Python, NumPy, PyTorch, CUDA, and deterministic flags where supported
  - last major interface change:
    - Node 20

- `strategies/step2_factor_residual_etf/run.sh`
  - active remote launch wrapper
  - auto-submits itself with `sbatch` when not already inside a Slurm allocation
  - presets:
    - `fast`
    - `scale_2_412M`
    - `scale_4_277M`
    - `stability_base`
    - `stability_dropout`
    - `stability_regularized`
  - performs a fast CUDA availability check before full project execution
  - large-model and stability presets currently use `batch_days = 16`

## Stability Map

Active high-impact files:

- `strategies/step2_factor_residual_etf/dataset.py`
- `model.py`
- `strategies/step1_sector_neutral/trainer.py`
- `strategies/step2_factor_residual_etf/main.py`
- `backtest.py`

Stable shared files:

- `config.py`
- `data_loader.py`
- `features.py`
- `loss.py`
- `visualization.py`
- `evaluate_saved_predictions.py`
- `utils.py`

Legacy / benchmark reference files:

- `dataset.py`
- `trainer.py`
- `main.py`
