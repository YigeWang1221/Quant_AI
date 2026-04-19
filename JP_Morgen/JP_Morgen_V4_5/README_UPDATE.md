# JP Morgan Quant V4.5 Update Plan

## Overview

This document summarizes the next optimization ideas for `JP_Morgen_V4_5`.

The current `Strategy_JPMorgen` baseline predicts raw 5-day forward returns and trains the model with a per-day cross-sectional ranking objective based on IC and ListNet-style loss. The main question for the next iteration is not whether the label is mathematically valid, but whether too much common market, sector, and factor exposure is still present in the target.

## Training Commands

The current training script supports both standard precision and forward-only mixed precision experiments.

Run examples:

```bash
python main.py
python main.py --amp_mode on --amp_dtype float16
python main.py --amp_mode on --amp_dtype bfloat16
python main.py --amp_mode off
```

Usage notes:

- `python main.py` uses the default configuration.
- `--amp_mode on --amp_dtype float16` enables CUDA mixed precision in the forward pass with `float16`.
- `--amp_mode on --amp_dtype bfloat16` enables CUDA mixed precision in the forward pass with `bfloat16`.
- `--amp_mode off` forces full precision training.
- The intended design is: forward pass may use mixed precision, while loss computation remains in `fp32` for better numerical stability.

## Current Conclusion

Cross-sectional demeaning of the target is conceptually reasonable, but in the current implementation it is expected to have very limited impact on training performance.

Why:

- The current target is the raw `pct_change(5).shift(-5)`.
- The IC loss already subtracts the daily cross-sectional mean before computing correlation.
- The ListNet objective is shift-invariant within each day because it applies `softmax(target / temperature)`.
- Loss is computed day by day, so adding or subtracting the same constant from all stocks on one date does not materially change the ranking objective.

Because of that, simple target demeaning is more of a semantic cleanup than a high-value experiment. It is unlikely to be the change that turns negative test IC into positive test IC.

## Main Hypothesis

The more important issue is that common return components are not removed aggressively enough. The model may still be learning broad market, sector, or style co-movements instead of stock-specific alpha.

That means the next iteration should focus on neutralizing shared exposures rather than only subtracting the daily cross-sectional mean.

## Strategy Candidates

### 1. Strategy_JPMorgen

Current baseline.

- Predict raw 5-day forward return.
- Train with daily IC/ListNet ranking loss.
- Keep this version fixed as the reference benchmark.

### 2. Strategy_MarketNeutralDemean

Use:

`y_i = r_i - mean_cross_section(r)`

Goal:

- Learn relative stock strength instead of market direction.

Assessment:

- Easy to implement.
- High feasibility.
- Low priority, because it is mostly redundant under the current loss design.

### 3. Strategy_SectorNeutral

Use:

`y_i = r_i - mean_sector(r)`

Goal:

- Remove sector and industry co-movement.
- Keep within-sector stock selection alpha.

Assessment:

- Low engineering cost.
- Can directly reuse the existing `STOCK_SECTOR_MAP`.
- High priority.
- Best lightweight experiment to run first.

### 4. Strategy_FactorResidual_ETF

Use a rolling regression or rolling beta framework with ETF proxies such as `SPY`, `QQQ`, `XLE`, `TLT`, and `GLD`, then define the label as:

`y_i = r_i - beta_i' f`

Goal:

- Remove broad systematic exposure from the future return label.
- Train on residual, more idiosyncratic alpha.

Assessment:

- Highest recommendation for the next major experiment.
- Stronger than simple demeaning because it removes exposure structure, not only average level.
- Feasible with current data because market ETF series already exist in the project.

### 5. Strategy_FactorResidual_FF5

Use Fama-French factor residuals instead of ETF proxies.

Goal:

- Build a cleaner academic factor-neutral target.

Assessment:

- Strong idea.
- Higher data and engineering cost than ETF residualization.
- Good follow-up experiment, but not the first one.

### 6. Strategy_ResidualMomentum

Residualize not only the future label but also return-based input features.

Goal:

- Reduce the chance that the model learns market and sector noise through momentum-style features.

Assessment:

- Medium implementation cost.
- More suitable after confirming that residual targets help.

### 7. Strategy_VolatilityManaged

Do not change the label first. Instead, scale prediction scores or portfolio weights by volatility at the portfolio construction stage.

Goal:

- Reduce noise from high-volatility names.
- Improve portfolio Sharpe rather than pure test IC.

Assessment:

- Useful for backtest improvement.
- Should be treated as a portfolio construction experiment, not a label-design experiment.

### 8. Strategy_TopBottomOrdinal

Replace continuous return labels with quantile or ordinal ranking labels.

Goal:

- Focus the model more directly on top/bottom separability.

Assessment:

- Possible, but requires larger changes to the head and/or loss.
- Medium priority.

## Recommended Experiment Order

1. Keep `Strategy_JPMorgen` as the fixed baseline.
2. Implement `Strategy_SectorNeutral`.
3. Implement `Strategy_FactorResidual_ETF`.
4. If residual labels help, test `Strategy_ResidualMomentum`.
5. Run `Strategy_VolatilityManaged` separately at the backtest layer.

## If Only One Optimization Is Chosen

The first choice should not be `Strategy_MarketNeutralDemean`.

The best single next experiment should be:

`Strategy_FactorResidual_ETF`

Reason:

- It removes systematic exposure more effectively than simple demeaning.
- It can be implemented using data already present in the project.
- It has the strongest chance of improving true stock-selection signal quality.

## Practical Implementation Notes

For the next version, the most actionable path is:

- Preserve the current baseline pipeline and results.
- Add a sector-neutral target option using the existing sector map.
- Add an ETF-residual target option using rolling beta estimation over a 60- to 120-day window.
- Compare all variants on validation IC, test IC, and long-short backtest metrics.
- Keep portfolio construction changes separate from target-engineering experiments.

## Planned Update Direction

The intended optimization direction for `JP_Morgen_V4_5` is:

- First test whether sector-neutral labeling improves cross-sectional stability.
- Then move to ETF-based factor residual labels as the primary next-generation target design.
- Only after target improvement is validated, consider residualized momentum features and volatility-managed portfolio construction.

This keeps the research path incremental, measurable, and aligned with the current architecture.

## Planned Training Loop Optimization

The next model-side optimization should focus on reducing Python-loop overhead in training rather than changing the strategy logic.

Current bottlenecks:

- Training still runs one forward pass per day inside a Python loop.
- Loss is also accumulated sample by sample inside Python loops.
- This limits GPU utilization even when CUDA mixed precision is enabled.

Planned improvement:

- Refactor the model to accept a fully batched tensor instead of running one day at a time.
- Convert the current per-day loop into a single batched forward call for each training batch.
- Keep padding and masks, but apply them in vectorized form across the batch.
- Preserve `fp32` loss computation after the batched predictions are produced.

Expected benefit:

- Better GPU utilization on the local RTX 4070 laptop GPU.
- More noticeable speedup than AMP alone.
- No change to the underlying strategy definition or training target.
