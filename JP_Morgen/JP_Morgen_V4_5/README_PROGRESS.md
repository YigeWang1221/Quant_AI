# JP Morgan Quant V4.5 Progress Tracker

Last updated: 2026-04-19

## Current Objective

Implement and evaluate Step 1 of the V4.5 optimization roadmap:

- `Strategy_SectorNeutral`
- Keep the original `main.py` baseline intact
- Run the new strategy from a dedicated folder
- Preserve clear experiment metadata in log outputs

## Active Strategy

Strategy code:

- `step1_sector_neutral`

Strategy summary:

- Predict 5-day forward returns after removing the same-day sector mean return
- Keep the model focused on within-sector stock selection
- Avoid changing baseline files for strategy logic

Key files:

- `strategies/step1_sector_neutral/main.py`
- `strategies/step1_sector_neutral/dataset.py`
- `strategies/step1_sector_neutral/trainer.py`
- `strategies/step1_sector_neutral/run.sh`

## Current Defaults

The active Step 1 defaults are intentionally aligned with the local `JP_Morgen_V4_MPS.ipynb` run profile that was stable on the user's machine:

- `d_model = 128`
- `num_layers = 3`
- `nhead = 4`
- `dropout = 0.15`
- `listnet_weight = 0.0`
- `lr = 3e-4`
- `batch_days = 24`
- `num_epochs = 100`
- `patience = 15`
- `amp_mode = off`

Trainer memory profile:

- Precompute training-day tensors on CPU
- Move only the current batch to GPU
- This differs from the baseline trainer, which preloads all training days onto GPU

## Logging Convention

Step 1 experiment outputs now include:

- A short run directory name in the form `V4_5__strategy-name__timestamp`
- A stdout log file whose filename matches the run name
- A stderr log file whose filename matches the run name
- `run_manifest.json` with structured metadata
- `run_manifest.txt` with a readable experiment header

The log header records:

- Strategy code and strategy summary
- Target definition
- Trainer profile
- Runtime device and AMP mode
- Loaded parameter values and their descriptions

## How To Run

From `JP_Morgen_V4_5/`:

```bash
python strategies/step1_sector_neutral/main.py
```

Windows example:

```powershell
& C:\Users\59386\.conda\envs\Quant311\python.exe D:\quant\quant1\JP_Morgen\JP_Morgen_V4_5\strategies\step1_sector_neutral\main.py
```

## Completed Changes

### 2026-04-19

- Implemented `Strategy_SectorNeutral` in a dedicated folder
- Replaced the first draft top-level Step 1 files with strategy-folder versions
- Switched Step 1 training to notebook-style local-memory loading to reduce OOM risk
- Updated Step 1 default parameters to match the user's local stable profile
- Added structured run manifests and richer log headers for Step 1 experiments

## Next Suggested Steps

- Run the Step 1 experiment and compare validation IC and test IC against the preserved baseline
- If Step 1 is stable, add a comparison script that summarizes baseline vs Step 1 runs
- After Step 1 evaluation, move to `Strategy_FactorResidual_ETF`
