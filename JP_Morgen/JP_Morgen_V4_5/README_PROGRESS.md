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

### 2026-04-20 | Node 09 | Controlled scale scripts created and first medium-scale run completed

Step:

- Create reusable local launch scripts for the active Step 2 branch and run the first medium-size expansion.

Problem:

- Capacity experiments need to be repeatable.
- The project needed a clean way to compare the current fast reference against larger models without editing arguments by hand every time.

Action:

- Add local PowerShell launchers:
  - `run_local_fast.ps1`
  - `run_local_scale_2_412M.ps1`
  - `run_local_scale_4_277M.ps1`
- Start the first expansion:
  - `d_model = 192`
  - `num_layers = 4`
  - `nhead = 6`
  - parameter count = `2,412,194` = `2.412M`
  - `batch_days = 20`
  - `amp_mode = on`

Observed result:

- Run: `V4_5__step2-factor-residual-etf__2026-04-20_0758`
- Completed successfully with no CUDA OOM
- `Avg Val IC = 0.0309`
- `Avg Test IC = 0.0070`
- `IC gap = 0.0239`
- Gross total return = `30.29%`
- Net total return = `-1.19%`
- Signal direction = `original`

Comparison versus the current fast Step 2 reference (`0.813M`):

- Parameter count:
  - `0.813M` -> `2.412M`
- Corrected `Avg Test IC`:
  - `0.0055` -> `0.0070`
- Corrected net total return:
  - `-8.75%` -> `-1.19%`
- `IC gap`:
  - `0.0174` -> `0.0239`

Interpretation:

- The first controlled capacity increase improved corrected test IC.
- It also improved corrected net return materially.
- The larger gap means overfitting risk increased, but the result is still strong enough to justify a second controlled scale test.

Expected effect:

- Use the `2.412M` result as the new reference point for deciding whether `4.277M` is worth running.

Status:

- Completed.

### 2026-04-20 | Node 10 | Remote 4.277M run became the provisional best model

Step:

- Review the remote Step 2 expansion result under the larger `256 / 4 / 8` configuration.

Problem:

- The `2.412M` model improved corrected test IC and net return, but its `IC gap` widened.
- It was still unclear whether more capacity would help further or simply overfit faster.

Action:

- Use the completed remote run:
  - `V4_5__step2-factor-residual-etf__2026-04-20_1003`
- Configuration:
  - `d_model = 256`
  - `num_layers = 4`
  - `nhead = 8`
  - parameter count = `4,277,122` = `4.277M`
  - `batch_days = 16`
  - `amp_mode = on`

Observed result:

- `Avg Val IC = 0.0262`
- `Avg Test IC = 0.0098`
- `IC gap = 0.0164`
- Gross total return = `44.33%`
- Net total return = `7.64%`
- Signal direction = `reversed`
- Positive test folds = `6 / 8`

Comparison across Step 2 model sizes:

- `0.813M`:
  - `Avg Test IC = 0.0055`
  - `Net Total = -8.75%`
  - `IC gap = 0.0174`
- `2.412M`:
  - `Avg Test IC = 0.0070`
  - `Net Total = -1.19%`
  - `IC gap = 0.0239`
- `4.277M`:
  - `Avg Test IC = 0.0098`
  - `Net Total = 7.64%`
  - `IC gap = 0.0164`

Interpretation:

- `4.277M` is the strongest Step 2 result seen so far.
- It is the first tested Step 2 size that turns corrected net return positive.
- It also avoids the `IC gap` deterioration seen at `2.412M`.
- However, yearly returns still flip by regime, so the result is promising but not yet fully trusted.

Expected effect:

- Treat `4.277M` as the provisional best model.
- Do not jump immediately to a larger model.
- First verify whether the `4.277M` gain is stable enough to reproduce.

Status:

- Completed.

### 2026-04-20 | Node 11 | Next-step decision after 4.277M

Step:

- Decide what to do after the first clearly positive Step 2 net result.

Problem:

- A larger model now looks helpful, but one strong run can still be luck, seed sensitivity, or regime alignment.

Decision:

- Do not scale further immediately.
- First run a robustness round around `4.277M`.

Recommended next actions:

1. Re-run `4.277M` at least once more with the same config to check repeatability.
2. Run a full `0.813M` fast-reference training under the new batched trainer so the smallest model is compared apples-to-apples with the new training path.
3. If `4.277M` remains the best after the repeat, start portfolio-construction tests on top of it:
   - larger `top_n`
   - lower `rebal_freq`
   - score smoothing
   - no-trade bands

Expected effect:

- Confirm whether the positive net result is robust before opening another capacity axis.

Status:

- Active next step.

### 2026-04-20 | Node 12 | Portfolio-construction layer added to Step 2

Step:

- Implement the first direct portfolio-construction upgrade on top of the Step 2 signal.

Problem:

- `4.277M` became the provisional best model, but the strategy still needed a better trading layer.
- The project needed direct support for:
  - larger baskets
  - score smoothing
  - no-trade bands
  - configurable rebalance spacing

Action:

- Extend `backtest.py` with:
  - score smoothing by ticker
  - cross-sectional signal normalization
  - incumbency-aware no-trade-band selection
- Extend `strategies/step2_factor_residual_etf/main.py` with:
  - `--score_smooth_window`
  - `--score_smooth_method`
  - `--no_trade_band`
- Save smoothed trade signals into `predictions.csv`
- Update cluster `run.sh` so the default preset is the `4.277M` model with the new portfolio-construction settings

Proxy calibration:

- A quick local sweep on the `2.412M` predictions suggested the strongest tested setting was:
  - `top_n = 5`
  - `rebal_freq = 5`
  - `score_smooth_method = ewm`
  - `score_smooth_window = 3`
  - `no_trade_band = 0.30`

Observed proxy effect on `2.412M` predictions:

- Old combo:
  - `top_n = 3`
  - `rebal_freq = 5`
  - no smoothing
  - no trade band
  - net total = `-0.90%`
- New combo:
  - `top_n = 5`
  - `rebal_freq = 5`
  - `ewm(3)` smoothing
  - `no_trade_band = 0.30`
  - net total = `30.92%`

Interpretation:

- Basket widening, smoothing, and a no-trade band materially improved the proxy result.
- In this proxy check, forcing `rebal_freq = 10` did not beat the stronger `rebal_freq = 5` combination, so the cluster default stayed at `5`.

Expected effect:

- The next `4.277M` cluster run will test the strongest current model with a more stable and less noisy portfolio-construction layer.

Status:

- Completed.

### 2026-04-20 | Node 13 | Cluster launch guard added

Step:

- Fix a cluster execution issue where the Step 2 run appeared to hang without finishing even one epoch.

Problem:

- The log showed:
  - `Device: cpu`
  - `AMP active: False`
  - `batch-loaded to cpu`
- This meant the job was not actually running on a GPU node.
- The most likely cause was launching `bash run.sh` on the login node, which ignores `#SBATCH` directives and runs the training directly on CPU.

Action:

- Update `strategies/step2_factor_residual_etf/run.sh` so that:
  - if no active Slurm allocation exists, the script auto-submits itself with `sbatch`
  - once inside a job, it prints hostname and CUDA visibility
  - it runs a fast PyTorch CUDA sanity check before any expensive data processing
  - it exits immediately with a clear error if CUDA is unavailable

Expected effect:

- Prevent long accidental CPU runs on the login node
- Make GPU allocation failures obvious in the first few seconds instead of after several minutes

Status:

- Completed.

## Active Defaults

Current active `step2` defaults:

- `d_model = 256`
- `num_layers = 4`
- `nhead = 8`
- `dropout = 0.15`
- `lr = 3e-4`
- `batch_days = 16`
- `num_epochs = 100`
- `patience = 15`
- `amp_mode = on`
- `amp_dtype = float16`
- `beta_window = 120`
- `beta_min_obs = 60`
- `top_n = 5`
- `rebal_freq = 5`
- `score_smooth_window = 3`
- `score_smooth_method = ewm`
- `no_trade_band = 0.30`

Current tested model sizes:

- Fast reference:
  - `128 / 3 / 4`
  - `812,738` params
  - `0.813M`
- Medium expansion:
  - `192 / 4 / 6`
  - `2,412,194` params
  - `2.412M`
- Current best tested expansion:
  - `256 / 4 / 8`
  - `4,277,122` params
  - `4.277M`

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
