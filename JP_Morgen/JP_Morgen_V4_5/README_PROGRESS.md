# JP Morgan Quant V4.5 Progress Timeline

Last updated: 2026-04-20

## Purpose And Maintenance Rules

Purpose:

- This file records the current working recommendation, stage-by-stage progress, and active defaults.
- Use it to answer "what should the next step be now?" rather than only "what changed historically?"

Maintenance rules:

- Keep the latest high-level recommendation at the top.
- Keep progress in chronological nodes so the reasoning stays readable by time and step.
- Progress nodes may include theory, interpretation, and stage conclusions, not only code changes.
- When the recommended next step changes, update both the top recommendation and the relevant timeline node.
- When a new validation tool becomes part of the workflow, record both why it is needed and where the file lives.

## Current Recommendation

Recommendation:

- Keep the active branch fixed at `Strategy_FactorResidual_ETF`.
- Keep corrected evaluation semantics fixed.
- Keep `4.277M` as the leading candidate size, but do not treat it as a stable default yet.
- Do not scale larger immediately.
- Do not promote the current Step 3 trading-layer preset to a trusted default yet.
- Keep running the same offline portfolio sweep on every saved `predictions.csv`.
- Shift the next work round toward stability debugging:
  - repeatability
  - seed sensitivity
  - signal-direction flips
  - regularization / validation discipline

Why this is the right time:

- The repeat round has now been executed on top of the `4.277M` size.
- The new completed repeat runs did not reproduce the strongest `1003` corrected IC result.
- The completed repeat runs also did not produce one robust trading-layer winner across runs.
- This means model size is still a reasonable candidate, but the stability gate has not been passed.

Recommended next experiments:

1. Keep evaluating every saved `4.277M` run under one common base trading layer:
   - `d_model = 256`
   - `num_layers = 4`
   - `nhead = 8`
   - `top_n = 3`
   - `rebal_freq = 5`
   - no smoothing
   - no trade band
2. Keep running the same offline sweep over every saved `predictions.csv`:
   - `top_n`
   - `rebal_freq`
   - `score_smooth_method`
   - `score_smooth_window`
   - `no_trade_band`
3. Open a stability-debug round before promoting Step 3:
   - compare seed-to-seed signal direction
   - test whether stronger regularization narrows the validation-to-test gap
   - investigate why `1003` remained strong while most later repeats weakened on corrected `Avg Test IC`

Promotion rule:

- Treat `4.277M` as a leading candidate size, not a locked stable default.
- Only promote a Step 3 portfolio-construction setting if it remains strong across multiple saved `predictions.csv`, not only on one lucky retrain.
- If the preferred signal direction keeps flipping across repeated `4.277M` runs, treat stability as the active blocker before opening another optimization axis.

## Repeat-Round Decision Table

Use this table after the two additional `4.277M` Step 2 reruns complete.

Comparison principle:

- Judge raw-alpha stability first with one common base trading layer:
  - `top_n = 3`
  - `rebal_freq = 5`
  - no smoothing
  - no trade band
- Then judge Step 3 readiness with the offline saved-prediction sweep.
- Do not mix one run's custom trading layer into another run's stability conclusion.

Reference anchors:

- Current corrected Step 2 fast reference:
  - `0.813M`
  - `Avg Test IC = 0.0055`
  - `Net Total = -8.75%`
- Current corrected Step 2 medium reference:
  - `2.412M`
  - `Avg Test IC = 0.0070`
  - `Net Total = -1.19%`
- Current strongest clean `4.277M` reference:
  - run `1003`
  - `Avg Test IC = 0.0098`
  - `Net Total = 7.64%`

Decision table:

| Pattern after repeat round | Interpretation | Next step |
|---|---|---|
| At least `2` repeated `4.277M` runs beat `0.813M` on corrected `Avg Test IC`, and the median repeated `Avg Test IC` is at or above the `2.412M` reference (`0.0070`) | Model size is repeatable enough to freeze at `4.277M` | Keep `4.277M` as the active model size and move into Step 3 portfolio optimization |
| Repeated `Avg Test IC` stays positive but remains mixed around the `2.412M` reference, while base-layer net results stay near flat or weakly negative | Model size is probably acceptable, but monetization is still the lever | Keep `4.277M`, start Step 3, and treat portfolio construction as the active optimization axis |
| Repeated runs keep flipping signal direction and at least two repeated runs fall below the `0.813M` Step 2 reference (`0.0055`) | The apparent `4.277M` edge is not stable enough yet | Do not open Step 3 fully; first investigate seed sensitivity, regularization, and train/validation instability |
| Base-layer corrected `Avg Test IC` looks stable, but net performance varies mainly after trading-layer changes | Alpha is more stable than monetization | Use `evaluate_saved_predictions.py` as the primary Step 3 gate and compare the same sweep grid on every run |
| A portfolio setting wins only on one run but not on the others in the offline sweep | That setting is probably run-specific rather than robust | Do not promote it to default; keep it as a candidate only |

Practical promotion rule:

- Freeze `4.277M` as the working size if the repeat round supports it on corrected `Avg Test IC`.
- Promote a Step 3 default only if the same portfolio configuration remains near the top across multiple saved runs.
- If size stability fails, fix repeatability first and postpone broader monetization claims.

Observed status after the current repeat round:

- The current repeat round did not pass the stability gate.
- Across the completed copied `4.277M` runs reviewed so far:
  - median corrected `Avg Test IC = 0.0032`
  - mean corrected `Avg Test IC = 0.0008`
  - only `1 / 5` completed runs met or exceeded the corrected `0.813M` Step 2 reference (`0.0055`)
  - only `1 / 5` completed runs met or exceeded the corrected `2.412M` reference (`0.0070`)
- On the common base trading layer:
  - only `1 / 5` completed runs finished with positive net return
- On the full offline sweep:
  - `3 / 5` completed runs could be made net positive
  - but the preferred direction and best configuration were not yet uniform enough to promote one stable Step 3 default

## Project Structure Snapshot

Use this section as a fast orientation map for future work.

Top-level files:

- `main.py`
  - baseline V4.5 entrypoint
- `dataset.py`
  - baseline target construction and normalized daily sample assembly
- `trainer.py`
  - baseline rolling-fold trainer path
- `model.py`
  - shared `QuantV4` model and `TwoWayBlock`
- `backtest.py`
  - shared signal preparation and long-short backtest logic
- `visualization.py`
  - shared plots for fold IC, strategy curve, daily IC, and rolling stock predictions
- `evaluate_saved_predictions.py`
  - offline portfolio sweep over saved `predictions.csv`
- `README_PROGRESS.md`
  - current recommendation, stage interpretation, defaults, and decision rules
- `README_UPDATE.md`
  - chronological implementation and decision log

Strategy folders:

- `strategies/step1_sector_neutral/`
  - Step 1 label and trainer entrypoint for sector-neutral target engineering
- `strategies/step2_factor_residual_etf/`
  - Step 2 label and trainer entrypoint for ETF-residual target engineering
  - current active branch for model-size and monetization work
  - `run.sh` now includes stability presets for repeatability debugging

Artifact folders:

- `log/`
  - local run outputs
- `update_img/`
  - supporting update images
- adjacent workspace folder `remote_result/V4_5`
  - copied remote cluster runs used for offline comparison

## Pipeline Snapshot

Baseline research pipeline:

1. `data_loader.py`
   - load cached market, stock, and fundamental data
2. `features.py`
   - compute engineered factor set
3. `dataset.py` or strategy-specific `dataset.py`
   - build target
   - normalize rolling factor windows
   - assemble `daily_samples`
4. `trainer.py` or strategy-specific `trainer.py`
   - create rolling folds
   - train `QuantV4`
   - derive per-fold seeds when a base `--seed` is supplied
   - save fold predictions
5. `main.py` or strategy-specific `main.py`
   - aggregate folds
   - choose signal direction
   - run backtest
   - save `predictions.csv`, `fold_metrics.csv`, `backtest.csv`
6. `visualization.py`
   - save research plots for review
7. `evaluate_saved_predictions.py`
   - run one common offline portfolio sweep on saved predictions

Active Step 2 pipeline:

1. `strategies/step2_factor_residual_etf/dataset.py`
   - compute ETF-residual forward-return label
2. `model.py`
   - run shared two-way transformer over time and cross-section
3. `strategies/step1_sector_neutral/trainer.py`
   - active fast batched trainer path reused by Step 2
4. `strategies/step2_factor_residual_etf/main.py`
   - active research entrypoint
   - now accepts:
     - `--seed`
     - `--deterministic`
     - `--weight_decay`
     - `--grad_clip_norm`
     - `--early_stop_min_delta`
5. `backtest.py`
   - convert predictions into tradable signals and evaluate long-short PnL
6. `evaluate_saved_predictions.py`
   - compare the same monetization grid across saved runs

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

### 2026-04-20 | Node 14 | Remote review showed 4.277M is large enough but not yet stable

Step:

- Compare the local corrected Step 2 runs with the copied remote `4.277M` runs under `remote_result/V4_5`.

Problem:

- `4.277M` already looked like the provisional best size, but it was still unclear whether that result was reproducible enough to treat as the new stable default.

Action:

- Compare the corrected local references:
  - `0.813M`
  - `2.412M`
- Compare the remote `4.277M` runs:
  - `2026-04-20_1003`
  - `2026-04-20_1728`
- Separate the model-size question from the monetization question by replaying old versus new portfolio settings on saved predictions.

Observed stage result:

- `4.277M` still sets the best corrected Step 2 size result seen so far:
  - `Avg Test IC = 0.0098`
  - `Net Total = 7.64%`
  - run `1003`
- But another same-size run with the portfolio-construction layer active produced:
  - `Avg Test IC = -0.0079`
  - `Net Total = 26.70%`
  - run `1728`
- This means same-size runs can still diverge materially in:
  - fold IC profile
  - signal direction
  - net monetization

Interpretation:

- The project has mostly answered "is the model still too small?"
- The active uncertainty is now "is `4.277M` repeatable enough to freeze as the working model size?"

Expected effect:

- Shift the next decision gate away from further scaling and toward repeatability testing.

Status:

- Completed.

### 2026-04-20 | Node 15 | Offline portfolio sweep became a required validation step

Step:

- Formalize offline portfolio-construction evaluation on saved `predictions.csv`.

Problem:

- The recent remote review showed that portfolio-construction gains can look strong on one run while the underlying corrected `Avg Test IC` is still unstable.
- Manual one-off checks are too easy to reproduce inconsistently.

Action:

- Add `evaluate_saved_predictions.py` at the project root.
- Make it run the same offline sweep over:
  - `top_n`
  - `rebal_freq`
  - `score_smooth_method`
  - `score_smooth_window`
  - `no_trade_band`
- Use identical sweep logic across multiple saved run folders or direct `predictions.csv` paths.

Interpretation:

- Step 3 should now mean "portfolio-construction optimization on a repeated prediction set," not "trust the strongest single run by default."

Expected effect:

- Reduce ad hoc analysis work.
- Keep the same monetization grid across repeated `4.277M` runs.
- Make the Step 3 gate easier to evaluate with less context rebuilding.

Status:

- Completed.

### 2026-04-20 | Node 16 | Additional 4.277M repeat runs failed the stability gate

Step:

- Review the additional remote `scale_4_277M` runs after the repeat round was expanded.

Problem:

- The project needed to know whether the earlier `4.277M` promise would reproduce once the same large model was launched several more times.

Action:

- Review copied remote runs:
  - `1003`
  - `1728`
  - `1756`
  - `1757`
  - `1811`
- Exclude `1649` as a failed CPU-path run because the log shows:
  - `Device: cpu`
  - `AMP active: False`
  - training started on CPU rather than on the intended GPU path
- Run the offline saved-prediction evaluation on the completed runs.

Observed stage result:

- Corrected `Avg Test IC` across the completed copied `4.277M` runs:
  - best = `0.0098` (`1003`)
  - median = `0.0032`
  - mean = `0.0008`
- Only `1 / 5` completed runs stayed at or above the corrected `0.813M` Step 2 reference.
- Only `1 / 5` completed runs stayed at or above the corrected `2.412M` reference.
- On the common base trading layer, only `1 / 5` completed runs finished net positive.
- On the full offline sweep, `3 / 5` completed runs could be made net positive, but:
  - the preferred direction still flipped
  - the best monetization setting still varied by run

Interpretation:

- `4.277M` is still a plausible leading model size.
- But the repeat round did not validate it as a stable default.
- The main blocker is now stability, not immediate further scaling or premature Step 3 promotion.

Expected effect:

- Redirect the next round away from "launch more scaling tiers" and toward "stability debugging plus continued offline sweep comparison."

Status:

- Completed.

### 2026-04-20 | Node 17 | Stability-experiment controls were added to the training path

Step:

- Add code support for reproducible stability debugging rather than relying on ad hoc reruns.

Problem:

- The project needed direct control over:
  - base random seed
  - per-fold reproducibility
  - stronger regularization
  - more selective early stopping
- Without these controls, the next stability round would still be difficult to interpret.

Action:

- Add shared seed control in `utils.py`
- Add new train-time arguments:
  - `--seed`
  - `--deterministic`
  - `--weight_decay`
  - `--grad_clip_norm`
  - `--early_stop_min_delta`
- Record `fold_seed` and `best_epoch` in fold metrics
- Update `strategies/step2_factor_residual_etf/run.sh` with stability presets:
  - `stability_base`
  - `stability_dropout`
  - `stability_regularized`

Expected effect:

- Make the next repeatability round cleaner to compare.
- Separate seed sensitivity from broader architecture questions.
- Make regularization tests easier to launch without rebuilding long command lines.

Status:

- Completed.

### 2026-04-21 | Node 18 | Seeded launch path was hotfixed after remote startup failure

Step:

- Repair the first seeded stability launches after they terminated before creating any internal run log.

Problem:

- The seed-enabled code path could fail during startup on the remote cluster before `setup_logging()` ran.
- The Slurm wrapper still printed environment diagnostics, which made the failure look like a silent stop.
- Root cause:
  - nested f-string syntax in `build_run_label()`
  - present in both active strategy entrypoints
  - parseable on newer local Python
  - incompatible with the remote Python 3.11 interpreter

Action:

- Rewrite run-label suffix construction to use plain variables:
  - `seed_suffix`
  - `deterministic_suffix`
- Keep the same naming semantics without nested f-strings.

Interpretation:

- That failure belongs to the launch path, not to model stability or GPU training behavior.
- Those failed seeded jobs should be excluded from any stability conclusion.

Expected effect:

- Seeded `stability_*` presets should now proceed into normal logging, artifact creation, and fold training on the cluster.

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
- Current leading tested expansion:
  - `256 / 4 / 8`
  - `4,277,122` params
  - `4.277M`
  - best seen one-off result so far, but not yet validated as a stable default

## Current File Anchors

- Active branch entrypoint:
  - `strategies/step2_factor_residual_etf/main.py`
- Active cluster launch presets:
  - `strategies/step2_factor_residual_etf/run.sh`
- Active fast trainer path:
  - `strategies/step1_sector_neutral/trainer.py`
  - `strategies/step2_factor_residual_etf/trainer.py`
- Shared model:
  - `model.py`
- Corrected backtest semantics:
  - `backtest.py`
  - `visualization.py`
- Offline portfolio sweep:
  - `evaluate_saved_predictions.py`
