# JP Morgan Quant V4.5 Progress Timeline

Last updated: 2026-04-20

## Current Recommendation

Recommendation:

- Start a controlled model-scaling round now instead of spending another full cycle on pre-analysis only.
- Keep the active branch fixed at `Strategy_FactorResidual_ETF`.
- Keep corrected evaluation semantics fixed.
- Scale capacity in small steps so we can tell whether extra model size helps out-of-sample IC without hiding behind implementation changes.

Why this is the right time:

- The evaluation bug has already been fixed.
- The active `step1` / `step2` trainer has already been sped up.
- CUDA AMP is now enabled by default on the active strategy branches.
- The current default model is still small at about `812,738` parameters.
- A small controlled capacity increase is now a cleaner experiment than it was before the trainer fix.

Recommended next experiments:

1. Keep the current fast reference:
   - `d_model = 128`
   - `num_layers = 3`
   - `nhead = 4`
   - `batch_days = 20`
   - `amp_mode = on`
2. Run a medium-size expansion:
   - `d_model = 192`
   - `num_layers = 4`
   - `nhead = 6`
   - start with `batch_days = 20`
   - if CUDA OOM appears, fall back to `batch_days = 16`
3. Run a larger but still practical expansion:
   - `d_model = 256`
   - `num_layers = 4`
   - `nhead = 8`
   - start with `batch_days = 16`

Promotion rule:

- Only keep a larger model if it improves corrected `Avg Test IC` and does not materially worsen `IC gap`.
- If a larger model raises validation metrics but weakens corrected test metrics, treat it as overfitting.
- If corrected gross returns improve but corrected net returns get worse, treat turnover and portfolio construction as the blocker rather than model size.

## Timeline

### 2026-04-19 | Node 01 | Baseline reference established

Step:

- Keep the original JPMorgan V4.5 pipeline as the benchmark.

Problem:

- The baseline predicts raw 5-day forward returns and mixes systematic and idiosyncratic effects.

Action:

- Preserve the baseline as the fixed comparison anchor.

Expected effect:

- All later strategy branches can be compared against one stable benchmark.

Status:

- Completed.

### 2026-04-19 | Node 02 | Step 1 sector-neutral target introduced

Step:

- Implement `Strategy_SectorNeutral` in its own strategy folder.

Problem:

- The model appeared to learn sector direction along with stock selection.

Action:

- Redefine the label as stock 5-day forward return minus same-day sector mean return.

Expected effect:

- Push the model toward within-sector stock ranking rather than sector beta.

Status:

- Completed.

### 2026-04-19 | Node 03 | Step 2 ETF-residual target introduced

Step:

- Implement `Strategy_FactorResidual_ETF` in its own strategy folder.

Problem:

- Sector-neutral labels remove sector average effects, but broad market and style exposure may still leak into the target.

Action:

- Estimate rolling ETF betas and subtract ETF-implied systematic forward return from the stock forward return label.

Expected effect:

- Train on a cleaner residual target with more idiosyncratic alpha content.

Status:

- Completed.

### 2026-04-20 | Node 04 | Evaluation semantics bug discovered

Step:

- Review the first baseline / step1 / step2 comparison after reading the saved prediction files and backtest path.

Problem:

- `predictions.csv` stored only transformed target values as `actual`.
- The backtest layer then used `actual` as realized return.
- This made baseline economically valid, but made Step 1 and Step 2 backtests incomparable to baseline.

Action:

- Preserve transformed `actual` for IC evaluation.
- Add `raw_actual` to predictions.
- Force backtest and realized-return plots to use raw forward return.

Expected effect:

- Restore one common economic backtest meaning across baseline, step1, and step2.

Status:

- Completed.

### 2026-04-20 | Node 05 | Corrected comparison rerun completed

Step:

- Re-run the corrected comparison set after the evaluation fix.

Problem:

- Earlier Step 1 / Step 2 PnL conclusions were no longer trustworthy after the evaluation bug was identified.

Action:

- Re-run baseline, step1, and step2 under corrected semantics.

Observed issue:

- Baseline default CUDA path OOMed when the preload trainer used `batch_days = 64`.

Solution:

- Re-run baseline safely with `--batch_days 24`.

Corrected result summary:

- Baseline:
  - `Avg Test IC = -0.0040`
  - `Net Total Return = -12.29%`
- Step 1:
  - `Avg Test IC = 0.0036`
  - `Net Total Return = -10.81%`
- Step 2:
  - `Avg Test IC = 0.0055`
  - `Net Total Return = -8.75%`

Expected effect:

- Establish the corrected strategy ranking before any new optimization axis is tested.

Status:

- Completed.

### 2026-04-20 | Node 06 | Trainer throughput bottleneck identified

Step:

- Diagnose why CUDA utilization stayed moderate even when VRAM was not full.

Problem:

- `batch_days` existed, but the trainer still ran one forward pass per day inside each batch.
- Validation and test inference also ran day by day.
- AMP was not active by default on the active strategy branches.

Action:

- Identify Python-loop overhead as the main avoidable bottleneck in the active Step 1 / Step 2 trainer.

Expected effect:

- Make the next speed improvement target precise before touching model size.

Status:

- Completed.

### 2026-04-20 | Node 07 | Trainer speed fix applied

Step:

- Speed up Step 1 / Step 2 training without increasing persistent GPU-memory pressure.

Problem:

- The active strategy trainer was underusing batch parallelism.

Action:

- Extend `QuantV4` and `TwoWayBlock` to accept batched day tensors.
- Replace Python-level per-day forward loops with one batched forward pass per batch.
- Batch validation and test inference.
- Keep daily tensors on CPU and move only the current batch to GPU.
- Add a vectorized IC-only loss path for the default `listnet_weight = 0.0` case.
- Turn AMP on by default for `step1` and `step2`.
- Reduce active default `batch_days` from `24` to `20` to keep activation memory conservative.

Validation:

- Smoke test run: `V4_5__step2-factor-residual-etf__2026-04-20_0744`
- Environment: `Quant311`, CUDA, `amp_mode = on`, `batch_days = 20`, `num_epochs = 2`
- Result: completed successfully with no CUDA OOM

Expected effect:

- Shorter epoch time.
- Better GPU utilization.
- No shift toward the baseline preload-to-GPU memory pattern.

Status:

- Completed.

### 2026-04-20 | Node 08 | Model-scaling decision

Step:

- Decide whether to scale the model now or delay scaling until after another analysis round.

Problem:

- The current model is small enough that it may underfit nonlinear factor interactions.
- At the same time, corrected net performance is still negative, so large uncontrolled expansion would risk adding overfit without fixing monetization.

Decision:

- Scale now, but only in a controlled way and only on the active `step2` branch.

Reasoning:

- This is now the cleanest point to test capacity because:
  - evaluation is corrected
  - trainer throughput is improved
  - AMP is enabled
  - memory pressure has been kept conservative

Expected effect:

- Learn whether moderate extra depth and width improve corrected out-of-sample IC.
- Avoid confusing capacity changes with evaluation bugs or trainer inefficiency.

Status:

- Active next step.

## Active Defaults

Current active `step2` defaults:

- `d_model = 128`
- `num_layers = 3`
- `nhead = 4`
- `dropout = 0.15`
- `lr = 3e-4`
- `batch_days = 20`
- `num_epochs = 100`
- `patience = 15`
- `amp_mode = on`
- `amp_dtype = float16`
- `beta_window = 120`
- `beta_min_obs = 60`

## Current File Anchors

- Active branch entrypoint:
  - `strategies/step2_factor_residual_etf/main.py`
- Active fast trainer path:
  - `strategies/step1_sector_neutral/trainer.py`
  - `strategies/step2_factor_residual_etf/trainer.py`
- Shared model:
  - `model.py`
- Corrected backtest semantics:
  - `backtest.py`
  - `visualization.py`
