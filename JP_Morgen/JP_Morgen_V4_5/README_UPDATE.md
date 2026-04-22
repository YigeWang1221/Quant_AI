# JP Morgan Quant V4.5 Update Timeline

Last updated: 2026-04-21

## Purpose

This file records the project as a timeline.

Each node should answer four questions:

- What step was taken?
- What problem or limitation triggered it?
- What solution was used?
- What effect was expected or observed?

This format is intentional. The project has already gone through one evaluation-semantics error, one baseline CUDA OOM, and one trainer-throughput refactor, so history needs to be readable in order rather than grouped only by topic.

## Maintenance Rules

- Keep this file strictly chronological.
- Add new nodes instead of silently rewriting the meaning of older results.
- Use this file for implementation changes, comparison decisions, and explicit next-step pivots.
- Use `README_PROGRESS.md` for the current recommendation, stage interpretation, and active working conclusions.
- Use `MODULES.md` for current module contracts, shapes, and output schemas.
- When a reusable validation tool is added, record both the motivation and the file path here.

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

### 2026-04-20 | Node 12 | Remote 4.277M result reviewed

Step:

- Compare the remote `4.277M` Step 2 run against the corrected smaller-model references.

Problem:

- The `2.412M` run improved corrected test IC, but its larger `IC gap` left open the question of whether more capacity would help or simply overfit.

Solution:

- Review the completed remote run:
  - `D:/quant/quant1/remote_result/V4_5/V4_5__step2-factor-residual-etf__2026-04-20_1003/`
- Use the full completed log from `1003` as the valid remote reference.
- Ignore `0911` as an incomplete copied result because its output stops before the summary.

Model size:

- `256 / 4 / 8`
- `4,277,122` params
- `4.277M`

Observed result:

- `Avg Val IC = 0.0262`
- `Avg Test IC = 0.0098`
- `IC gap = 0.0164`
- Gross total return = `44.33%`
- Net total return = `7.64%`
- Signal direction = `reversed`
- Positive test folds = `6 / 8`

Comparison versus earlier Step 2 sizes:

- `0.813M`
  - `Avg Test IC = 0.0055`
  - `Net Total = -8.75%`
  - `IC gap = 0.0174`
- `2.412M`
  - `Avg Test IC = 0.0070`
  - `Net Total = -1.19%`
  - `IC gap = 0.0239`
- `4.277M`
  - `Avg Test IC = 0.0098`
  - `Net Total = 7.64%`
  - `IC gap = 0.0164`

Interpretation:

- `4.277M` is the best Step 2 run so far.
- It is the first tested Step 2 size with positive corrected net return.
- It improved corrected test IC further while also pulling the generalization gap back down relative to `2.412M`.
- This makes the result materially more interesting than a simple "bigger model overfits harder" story.

Remaining caution:

- Mean daily IC is still negative in that run.
- Year-by-year returns still show regime sensitivity.
- So the result is promising, but it should still be validated before treating it as the new final default.

Status:

- Completed.

### 2026-04-20 | Node 13 | Next-step decision after 4.277M

Step:

- Decide whether to keep scaling model size or pivot to validation and monetization work.

Decision:

- Do not push to a larger model immediately.
- First validate `4.277M` and make the comparison cleaner.

Recommended next actions:

1. Re-run `4.277M` once more with the same config to test repeatability.
2. Run a full `0.813M` training under the new fast trainer so the smallest model is compared apples-to-apples with the new batched path.
3. If `4.277M` remains the best after the repeat, start portfolio-construction tests on top of it rather than opening another capacity tier first.

Reason:

- The new result is strong enough to justify serious follow-up.
- It is not yet strong enough to justify blindly scaling further without stability checks.

Status:

- Active next step.

### 2026-04-20 | Node 14 | Portfolio-construction upgrade implemented

Step:

- Turn the earlier portfolio-construction ideas into actual Step 2 code and cluster defaults.

Problem:

- The model side had improved, but the trading layer still needed direct control over:
  - basket width
  - score smoothing
  - incumbency / no-trade behavior
  - rebalance spacing

Solution:

- Extend `backtest.py` so Step 2 can trade on a prepared signal instead of raw predictions only.
- Add:
  - score smoothing by ticker
  - normalized trade signals
  - no-trade-band selection logic
- Extend `strategies/step2_factor_residual_etf/main.py` with:
  - `--score_smooth_window`
  - `--score_smooth_method`
  - `--no_trade_band`
- Save trade-signal columns into `predictions.csv`
- Update `strategies/step2_factor_residual_etf/run.sh` so the cluster default is now:
  - `4.277M`
  - `top_n = 5`
  - `rebal_freq = 5`
  - `score_smooth_method = ewm`
  - `score_smooth_window = 3`
  - `no_trade_band = 0.30`

Proxy calibration result:

- A quick local sweep on the `2.412M` predictions showed a strong result for:
  - `top_n = 5`
  - `rebal_freq = 5`
  - `ewm(3)` smoothing
  - `no_trade_band = 0.30`

Observed proxy effect on `2.412M` predictions:

- Old combo:
  - net total = `-0.90%`
  - avg turnover = `0.7030`
- New combo:
  - net total = `30.92%`
  - avg turnover = `0.5558`

Interpretation:

- Basket widening plus smoothing plus an incumbency band materially changed the monetization layer.
- A lower trading frequency remained available as a control, but in this proxy check `rebal_freq = 10` was not the strongest tested setting, so it was not made the default.

Status:

- Completed.

### 2026-04-20 | Node 15 | Cluster CPU-launch issue fixed

Step:

- Diagnose why a supposedly GPU-backed Step 2 cluster run spent a long time without finishing even one epoch.

Observed symptom:

- The run log showed:
  - `Device: cpu`
  - `AMP active: False`
  - `batch-loaded to cpu`
- The stderr log stayed empty.

Interpretation:

- The code was not stuck on a V100 or H200 GPU.
- It was running on the login node CPU instead.
- This happens when `bash run.sh` is executed directly on the login node, because `#SBATCH` directives are only used by `sbatch`.

Solution:

- Update `strategies/step2_factor_residual_etf/run.sh` so that:
  - if there is no active Slurm allocation, it auto-submits itself with `sbatch`
  - once inside the job, it prints host and CUDA diagnostics
  - it performs a fast `torch.cuda.is_available()` check before full data loading
  - it exits immediately with a clear error if no GPU is actually visible

Expected effect:

- No more silent long CPU runs on the login node
- Faster diagnosis when a GPU allocation fails or lands on the wrong environment

Status:

- Completed.

### 2026-04-20 | Node 16 | Remote 4.277M review reframed the next-step decision

Step:

- Compare the local corrected Step 2 reference runs with the copied remote `4.277M` runs under `remote_result/V4_5`.

Problem:

- `4.277M` had become the provisional best model size, but the project still needed to answer two different questions cleanly:
  - is the model-size question now mostly solved?
  - is the result stable enough to move on to Step 3-style monetization work?

Solution:

- Compare:
  - local corrected `0.813M`
  - local corrected `2.412M`
  - remote `4.277M` run `1003`
  - remote `4.277M` run `1728`
- Replay old and new trading-layer settings on saved predictions to separate model-size improvement from portfolio-construction improvement.

Observed result:

- `1003` remained the strongest clean size-scaling result:
  - `Avg Test IC = 0.0098`
  - `Net Total = 7.64%`
  - signal direction = `reversed`
- `1728` produced a much stronger monetization result:
  - `Avg Test IC = -0.0079`
  - `Net Total = 26.70%`
  - signal direction = `original`
- Replaying the same portfolio settings on saved predictions showed:
  - `1003` still preferred the old base trading layer over the new one
  - `1728` strongly preferred the new trading layer over the old one

Interpretation:

- The project has mostly answered the model-size question:
  - `4.277M` is large enough to stop the immediate scaling search
- But it has not yet answered the repeatability question:
  - same-size runs still differ materially in corrected IC profile, monetization, and preferred signal direction

Decision:

- Freeze the next repeat round at `4.277M`.
- Do not scale larger yet.
- Require at least two more same-config Step 2 reruns before treating `4.277M` as the stable default for Step 3 optimization.

Expected effect:

- Move the next validation gate from "is bigger better?" to "is the chosen size repeatable enough to optimize around?"

Status:

- Completed.

### 2026-04-20 | Node 17 | Offline saved-prediction sweep tool added

Step:

- Add a reusable offline evaluation script for saved `predictions.csv`.

Problem:

- The project needed one consistent way to re-test portfolio-construction settings across multiple saved runs without redoing manual ad hoc analysis.
- The remote `4.277M` review showed that this comparison now matters directly for the Step 3 gate.

Solution:

- Add `evaluate_saved_predictions.py` at the project root.
- Make it accept:
  - one run directory
  - a directory tree of runs
  - or one or more direct `predictions.csv` files
- Make it sweep:
  - `top_n`
  - `rebal_freq`
  - `score_smooth_method`
  - `score_smooth_window`
  - `no_trade_band`
- Save:
  - `sweep_summary.csv`
  - `best_config_by_run.csv`
  - `best_config_yearly.csv`
  - `sweep_manifest.json`

Expected effect:

- Keep the monetization grid identical across repeated saved runs.
- Reduce context-heavy manual comparison work.
- Make the Step 3 portfolio-construction decision easier to reproduce after each new `4.277M` rerun.

Status:

- Completed.

### 2026-04-20 | Node 18 | Readme received a fixed repeat-round gate and fast context snapshot

Step:

- Upgrade the project readme layer so future conversations can recover context faster.

Problem:

- The project now has enough moving parts that ad hoc memory is too expensive:
  - baseline versus Step 1 versus Step 2
  - size scaling versus monetization
  - local runs versus copied remote runs
- The repeat round also needed a fixed decision table instead of a verbal one-off recommendation.

Solution:

- Extend `README_PROGRESS.md` with:
  - a fixed `4.277M` repeat-round decision table
  - a project-structure snapshot
  - a pipeline snapshot
- Keep `README_UPDATE.md` chronological and record this documentation upgrade explicitly.

Expected effect:

- Make future analysis turns faster to restart.
- Reduce repeated explanation of where key files live and how a saved run should be judged.
- Keep the Step 3 gate readable without rebuilding the whole reasoning chain each time.

Status:

- Completed.

### 2026-04-20 | Node 19 | Expanded 4.277M repeat round did not confirm stability

Step:

- Review the newly copied remote runs after several more `sbatch ./run.sh scale_4_277M` launches.

Problem:

- The project needed a real repeat-round answer, not just one strong `4.277M` run and one monetization-heavy follow-up.

Solution:

- Review completed copied runs:
  - `1003`
  - `1728`
  - `1756`
  - `1757`
  - `1811`
- Treat `1649` as invalid for model comparison because it ran on CPU:
  - `Device: cpu`
  - `AMP active: False`
- Evaluate the completed runs in two ways:
  - one common base trading layer for raw-alpha comparison
  - the offline saved-prediction sweep for Step 3-style monetization comparison

Observed result:

- Corrected `Avg Test IC` across the completed copied `4.277M` runs:
  - best = `0.0098`
  - median = `0.0032`
  - mean = `0.0008`
- Only `1 / 5` completed runs reached or exceeded the corrected `0.813M` Step 2 reference.
- Only `1 / 5` completed runs reached or exceeded the corrected `2.412M` reference.
- Under the common base trading layer:
  - only `1 / 5` completed runs were net positive
- Under the offline sweep:
  - `3 / 5` completed runs could be made net positive
  - but the preferred signal direction and best monetization settings still varied by run

Interpretation:

- The expanded repeat round did not confirm `4.277M` as a stable default.
- The size is still a plausible leading candidate, but the project cannot yet claim repeatable raw-alpha strength at that size.
- The right next move is stability work, not immediate larger scaling or blind Step 3 promotion.

Status:

- Completed.

### 2026-04-20 | Node 20 | Stability-round training controls added

Step:

- Add code support for a more disciplined repeatability round.

Problem:

- The project had identified stability as the next blocker, but the active code path still lacked direct controls for:
  - explicit seeds
  - per-fold reproducibility
  - stronger regularization
  - more selective early stopping

Solution:

- Add shared seed control in `utils.py`
- Add new CLI arguments to the Step 1 / Step 2 active trainer path:
  - `--seed`
  - `--deterministic`
  - `--weight_decay`
  - `--grad_clip_norm`
  - `--early_stop_min_delta`
- Update the shared Step 1 trainer implementation to:
  - derive per-fold seeds from the base seed
  - log the fold seed
  - record `best_epoch`
  - use configurable weight decay and gradient clipping
- Update `strategies/step2_factor_residual_etf/run.sh` with stability presets:
  - `stability_base`
  - `stability_dropout`
  - `stability_regularized`

Expected effect:

- Make the next stability round easier to interpret.
- Let the project compare seed sensitivity and regularization changes without rebuilding commands by hand.
- Reduce confusion between one-off randomness and real model-quality changes.

Status:

- Completed.

### 2026-04-21 | Node 21 | Seeded launch path hotfix applied for remote Python 3.11

Step:

- Fix the first seeded stability launches after they stopped before any project log file was created.

Problem:

- The new seed-enabled launch path could exit before `setup_logging()` initialized the run directory.
- That made the Slurm job print the outer environment checks but create no internal project log.
- Root cause:
  - nested f-string syntax inside `build_run_label()`
  - used in:
    - `strategies/step2_factor_residual_etf/main.py`
    - `strategies/step1_sector_neutral/main.py`
  - accepted by newer local Python parsing
  - not accepted by the remote Python 3.11 interpreter

Solution:

- Replace the nested f-string suffix construction with plain temporary variables:
  - `seed_suffix`
  - `deterministic_suffix`
- Keep the same run-label semantics while removing the Python-version compatibility hazard.

Interpretation:

- The failed seeded launches were startup compatibility failures, not trainer hangs.
- Missing project-side log files are expected for that broken version because the process died before the logging path was created.

Expected effect:

- Seeded `stability_*` presets should now reach normal project logging and artifact creation on the remote cluster.

Status:

- Completed.

### 2026-04-21 | Node 22 | Module-contract layer and AI prompt scaffolds added

Step:

- Add a dedicated contract snapshot for current module wiring and reusable prompt files for future AI sessions.

Problem:

- `README_PROGRESS.md` and `README_UPDATE.md` already explained project history and current recommendation, but they still did not answer:
  - what each module expects as input now
  - which trainer path is active
  - which CSV columns are safe to treat as economic returns
  - which files should be loaded for each task type
- That kept forcing new AI conversations to read code just to reconstruct stable interfaces.

Solution:

- Add `MODULES.md` as the L2 module-contract layer.
- Record:
  - tensor shapes
  - `full_data` structure
  - output artifact schemas
  - active dependencies
  - stability status and last major change point by module
- Add `AGENTS.md` and `CLAUDE.md` so future Codex / Claude sessions start from the same file-scope and task-template rules.
- Update `README_PROGRESS.md` so L1 now explicitly points to `MODULES.md`.

Files touched:

- `MODULES.md`
- `AGENTS.md`
- `CLAUDE.md`
- `README_PROGRESS.md`
- `README_UPDATE.md`

Expected effect:

- Faster context recovery in future AI sessions.
- Less repeated code-reading for questions about interfaces and dependencies.
- Narrower, less diffuse conversations because task templates and writable-file scope are now documented.

Status:

- Completed.

### 2026-04-21 | Node 23 | Agent guidance was collapsed back to one source of truth

Step:

- Rework the mixed Codex + Claude Code docs so the project does not maintain two independent agent-policy documents.

Problem:

- A split `AGENTS.md` / `CLAUDE.md` policy setup invites drift.
- The project needed Claude compatibility without letting prompt rules fork into separate maintenance tracks.

Solution:

- Keep `AGENTS.md` as the single authoritative agent-policy file.
- Add `PROJECT_KNOWLEDGE.md` as the compact shared knowledge file intended for mixed Codex + Claude Code use.
- Reduce `CLAUDE.md` to a compatibility note that points back to `AGENTS.md`.

Files touched:

- `AGENTS.md`
- `PROJECT_KNOWLEDGE.md`
- `CLAUDE.md`
- `README_PROGRESS.md`
- `README_UPDATE.md`

Expected effect:

- One policy file instead of two drifting policy files.
- One compact knowledge file that can be attached to project knowledge or used as a shared warm-start context.
- Cleaner maintenance when the project workflow changes later.

Status:

- Completed.

## Current Operating Defaults

Active Step 2 defaults:

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

Conclusion 7:

- `4.277M` answered that question positively enough to become the provisional best model, so the next priority is robustness confirmation, not immediate further scaling.

Conclusion 8:

- The next serious Step 2 cluster run should use the `4.277M` model together with the new portfolio-construction layer, because signal monetization is now the most leveraged remaining optimization axis.

Conclusion 9:

- A cluster run that reports `Device: cpu` is an execution-path issue, not evidence that the trainer itself is hanging on GPU.

Conclusion 10:

- `4.277M` is now large enough to freeze for the next repeat round; the active blocker is stability, not immediate further scaling.

Conclusion 11:

- Step 3 should now be treated as a gated portfolio-construction phase that runs on repeated saved predictions, not as an automatic next step after one strong run.

Conclusion 12:

- The expanded `4.277M` repeat round did not pass the stability gate, so current priorities should shift toward repeatability debugging and more disciplined promotion criteria.

Conclusion 13:

- The project now has the code-level controls needed for a real stability-debug round, so the next experiments should use explicit seeds and named stability presets rather than generic repeated reruns.

Conclusion 14:

- The first failed seeded launches should not be counted as evidence about stability because they were caused by a startup Python-version compatibility bug that has now been patched.
