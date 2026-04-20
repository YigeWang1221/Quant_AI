# JP Morgan Quant V4.5 Update Timeline

Last updated: 2026-04-20

## Purpose

This file records the project as a timeline.

Each node should answer four questions:

- What step was taken?
- What problem or limitation triggered it?
- What solution was used?
- What effect was expected or observed?

This format is intentional. The project has already gone through one evaluation-semantics error, one baseline CUDA OOM, and one trainer-throughput refactor, so history needs to be readable in order rather than grouped only by topic.

## Timeline

### 2026-04-19 | Node 01 | Baseline kept fixed

Step:

- Preserve the original JPMorgan V4.5 pipeline as the benchmark branch.

Problem:

- Later target-engineering experiments need one stable comparison point.

Solution:

- Keep the baseline target as raw 5-day forward return.
- Keep the original baseline architecture and training objective as the reference implementation.

Expected effect:

- All strategy variants can be compared to one consistent benchmark.

Status:

- Completed.

### 2026-04-19 | Node 02 | Step 1 sector-neutral strategy implemented

Step:

- Create `strategies/step1_sector_neutral/`.

Problem:

- The original target can reward sector direction instead of pure stock selection.

Solution:

- Define the Step 1 label as:
  - stock 5-day forward return
  - minus same-day sector mean 5-day forward return
- Keep the strategy isolated in its own folder so baseline files remain unchanged.

Expected effect:

- Push the model toward within-sector relative ranking.

Status:

- Completed.

### 2026-04-19 | Node 03 | Step 2 ETF-residual strategy implemented

Step:

- Create `strategies/step2_factor_residual_etf/`.

Problem:

- Sector-neutral labels remove only one layer of shared movement.
- Market, style, and ETF-proxy exposures may still dominate the target.

Solution:

- Estimate trailing stock-to-ETF betas from daily return history.
- Define the Step 2 label as:
  - raw stock 5-day forward return
  - minus ETF-implied systematic 5-day forward return
- Use proxies:
  - `SPY`
  - `QQQ`
  - `XLE`
  - `TLT`
  - `GLD`

Expected effect:

- Create a more idiosyncratic label than simple sector demeaning.

Status:

- Completed.

### 2026-04-19 to 2026-04-20 | Node 04 | First comparison produced misleading conclusions

Step:

- Review the first baseline / step1 / step2 comparison.

Observed first-pass interpretation:

- Step 1 looked much better than baseline.
- Step 1 seemed to work with the original signal direction.
- Step 2 looked highly regime-sensitive.

Problem:

- Those conclusions were based on a hidden evaluation mismatch.

Root cause:

- `predictions.csv` stored only the training target in the `actual` column.
- The backtest layer then used `actual` as realized return.
- This was valid for baseline because baseline target equals raw forward return.
- This was invalid for Step 1 and Step 2 because their targets are transformed labels.

Impact:

- Baseline PnL remained economically meaningful.
- Step 1 and Step 2 PnL were not directly comparable to baseline.
- Signal-direction conclusions drawn from those PnL numbers became unreliable.

Expected effect of fixing it:

- One common realized-return meaning across all strategy branches.

Status:

- Historical issue identified and superseded.

### 2026-04-20 | Node 05 | Evaluation semantics fixed

Step:

- Correct the prediction-output and backtest semantics.

Problem:

- The project needed to preserve both:
  - transformed training labels for IC evaluation
  - raw realized returns for economic backtesting

Solution:

- Keep transformed `actual` in prediction outputs for target-space IC evaluation.
- Add `raw_actual` to prediction outputs.
- Make backtest and realized-return plots use `raw_actual` when available.

Files touched:

- `dataset.py`
- `trainer.py`
- `strategies/step1_sector_neutral/dataset.py`
- `strategies/step1_sector_neutral/trainer.py`
- `strategies/step2_factor_residual_etf/dataset.py`
- `backtest.py`
- `visualization.py`

Expected effect:

- IC answers:
  - "did the model rank the chosen target well?"
- Backtest answers:
  - "did the model make money on raw forward returns?"

Status:

- Completed.

### 2026-04-20 | Node 06 | Baseline corrected rerun hit CUDA OOM

Step:

- Re-run the benchmark after the evaluation fix.

Problem:

- The baseline trainer preloads all training-day tensors onto GPU.
- The default baseline configuration used `batch_days = 64`.
- This combination exceeded the safe memory range on the local RTX 4070 laptop GPU.

Observed failure:

- Run: `V4_5__baseline-jpmorgen__2026-04-20_0027`
- Error: `torch.cuda.OutOfMemoryError`

Solution:

- Keep the baseline trainer logic unchanged.
- Re-run baseline safely with `--batch_days 24`.

Successful baseline rerun:

- Run: `V4_5__baseline-jpmorgen__2026-04-20_0028`

Expected effect:

- Preserve a valid corrected baseline comparison without rewriting the baseline trainer.

Status:

- Completed.

### 2026-04-20 | Node 07 | Corrected comparison established

Step:

- Compare baseline, Step 1, and Step 2 again after the evaluation fix.

Reference runs:

- Baseline safe rerun:
  - `V4_5__baseline-jpmorgen__2026-04-20_0028`
- Step 1 rerun:
  - `V4_5__step1-sector-neutral__2026-04-20_0120`
- Step 2 rerun:
  - `V4_5__step2-factor-residual-etf__2026-04-20_0219`

Corrected result summary:

- Baseline:
  - `Avg Val IC = 0.0102`
  - `Avg Test IC = -0.0040`
  - `IC gap = 0.0142`
  - Gross total return = `13.85%`
  - Net total return = `-12.29%`
  - Signal direction = `reversed`
- Step 1:
  - `Avg Val IC = 0.0208`
  - `Avg Test IC = 0.0036`
  - `IC gap = 0.0172`
  - Gross total return = `17.39%`
  - Net total return = `-10.81%`
  - Signal direction = `reversed`
- Step 2:
  - `Avg Val IC = 0.0229`
  - `Avg Test IC = 0.0055`
  - `IC gap = 0.0174`
  - Gross total return = `19.99%`
  - Net total return = `-8.75%`
  - Signal direction = `reversed`

Additional implementation signal:

- Baseline average turnover ~ `0.662`
- Step 1 average turnover ~ `0.696`
- Step 2 average turnover ~ `0.695`

Interpretation:

- Step 2 became the best target-design branch under corrected evaluation.
- Step 1 remained better than the corrected baseline.
- All three strategies still lost money net of cost.
- The remaining blocker looked more like monetization / turnover than raw model size alone.

Expected effect:

- Freeze the corrected strategy ranking before opening a new optimization axis.

Status:

- Completed.

### 2026-04-20 | Node 08 | Trainer throughput bottleneck diagnosed

Step:

- Review CUDA utilization and active trainer structure.

Problem:

- CUDA was not fully utilized even when VRAM was not full.
- The active Step 1 / Step 2 trainer already had `batch_days`, but still executed:
  - one forward pass per day inside each batch
  - one day at a time during validation
  - one day at a time during test inference

Interpretation:

- The main avoidable bottleneck was Python-loop overhead, not lack of VRAM alone.

Expected effect of addressing it:

- Better throughput without changing the corrected evaluation semantics.

Status:

- Completed.

### 2026-04-20 | Node 09 | Trainer speed fix applied

Step:

- Speed up the active Step 1 / Step 2 trainer without shifting toward a more VRAM-heavy baseline-style preload path.

Problem:

- The current strategy trainer underused batch parallelism.

Solution:

- Extend `QuantV4` and `TwoWayBlock` so they accept batched day tensors.
- Run one batched forward pass per date batch instead of looping day by day.
- Batch validation and test inference by date groups.
- Keep daily tensors on CPU and move only the current batch to GPU.
- Add a vectorized IC-only loss path for the default `listnet_weight = 0.0` case.
- Turn CUDA AMP on by default for `step1` and `step2`.
- Reduce active default `batch_days` from `24` to `20` to avoid pushing activation memory too high.

Files touched:

- `model.py`
- `config.py`
- `strategies/step1_sector_neutral/main.py`
- `strategies/step1_sector_neutral/trainer.py`
- `strategies/step2_factor_residual_etf/main.py`

Validation:

- `py_compile` passed on the modified files.
- Vectorized IC-only loss matched the previous loop implementation in a direct numerical check.

Smoke test:

- Run: `V4_5__step2-factor-residual-etf__2026-04-20_0744`
- Environment: `Quant311`
- Parameters:
  - `num_epochs = 2`
  - `batch_days = 20`
  - `amp_mode = on`
- Result:
  - completed successfully
  - no CUDA OOM

Expected effect:

- Lower epoch time.
- Better effective GPU usage.
- No material increase in persistent GPU memory pressure.

Status:

- Completed.

### 2026-04-20 | Node 10 | Capacity-scaling decision

Step:

- Decide whether to scale the model now or postpone scaling until after more analysis.

Problem:

- The current model is still small:
  - `d_model = 128`
  - `num_layers = 3`
  - `nhead = 4`
  - about `812,738` parameters
- This size may be too small to capture richer nonlinear interactions between factors.
- But corrected net returns are still negative, so aggressive scaling would risk overfit without solving monetization.

Decision:

- Start scaling now, but do it as a controlled experiment rather than a full strategy rewrite.

Why now:

- evaluation semantics are already corrected
- active trainer overhead has already been reduced
- AMP is on
- the memory profile has already been made more conservative

Recommended order:

1. Keep the current fast Step 2 reference:
   - `d_model = 128`
   - `num_layers = 3`
   - `nhead = 4`
   - `batch_days = 20`
   - `amp_mode = on`
2. First expansion:
   - `d_model = 192`
   - `num_layers = 4`
   - `nhead = 6`
   - start with `batch_days = 20`
   - if OOM, drop to `16`
3. Second expansion:
   - `d_model = 256`
   - `num_layers = 4`
   - `nhead = 8`
   - start with `batch_days = 16`

Promotion rule:

- Keep a larger model only if:
  - corrected `Avg Test IC` improves
  - corrected `IC gap` does not widen too much
  - corrected net return does not deteriorate

Expected effect:

- Test whether moderate extra capacity helps model nonlinear factor structure without mixing in unrelated system changes.

Status:

- Active next step.

### 2026-04-20 | Node 11 | First controlled capacity expansion executed

Step:

- Turn the model-scaling recommendation into reusable local launchers and run the first medium-size Step 2 expansion.

Problem:

- Capacity experiments should be repeatable and easy to compare.
- Manual CLI editing makes it too easy to lose track of which exact model size produced which run.

Solution:

- Add local launchers:
  - `strategies/step2_factor_residual_etf/run_local_fast.ps1`
  - `strategies/step2_factor_residual_etf/run_local_scale_2_412M.ps1`
  - `strategies/step2_factor_residual_etf/run_local_scale_4_277M.ps1`
- Use the first expansion config:
  - `d_model = 192`
  - `num_layers = 4`
  - `nhead = 6`
  - `batch_days = 20`
  - `amp_mode = on`

Model-size note:

- Current fast reference:
  - `812,738` params
  - `0.813M`
- First expansion:
  - `2,412,194` params
  - `2.412M`
- Second prepared expansion:
  - `4,277,122` params
  - `4.277M`

Run result:

- Run: `V4_5__step2-factor-residual-etf__2026-04-20_0758`
- Result: completed successfully, no CUDA OOM
- `Avg Val IC = 0.0309`
- `Avg Test IC = 0.0070`
- `IC gap = 0.0239`
- Gross total return = `30.29%`
- Net total return = `-1.19%`
- Signal direction = `original`

Comparison versus the corrected fast Step 2 reference:

- `Avg Test IC`:
  - `0.0055` -> `0.0070`
- Net total return:
  - `-8.75%` -> `-1.19%`
- `IC gap`:
  - `0.0174` -> `0.0239`

Interpretation:

- The first controlled expansion improved corrected out-of-sample IC.
- It also improved corrected net return materially.
- The larger validation-to-test gap means overfit risk rose, but not enough to reject the expansion.

Expected next effect:

- Use `2.412M` as the new comparison point for deciding whether to run `4.277M`.

Status:

- Completed.

## Current Operating Defaults

Active Step 2 defaults:

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

Prepared local scale presets:

- Fast reference:
  - `run_local_fast.ps1`
  - `0.813M`
- First expansion:
  - `run_local_scale_2_412M.ps1`
  - `2.412M`
- Second expansion:
  - `run_local_scale_4_277M.ps1`
  - `4.277M`

## Current Working Conclusions

Conclusion 1:

- Step 2 is the best target-design branch tested so far under corrected evaluation.

Conclusion 2:

- The earlier Step 1 "original signal works" story belonged to the buggy evaluation stage and should be treated as historical only.

Conclusion 3:

- Trainer speed was worth fixing before scaling the model because the old Python-loop path was masking how much compute was actually available.

Conclusion 4:

- Now that evaluation and throughput are both cleaner, moderate model scaling is reasonable.

Conclusion 5:

- Model scaling should still be treated as a controlled experiment, not as a guaranteed solution to the net-return problem.

Conclusion 6:

- The first controlled scale-up to `2.412M` is promising enough that the next logical question is whether `4.277M` keeps improving corrected test IC without widening the generalization gap too much.
