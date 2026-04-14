import numpy as np
import pandas as pd


def compute_v3_factors(df, market_data, fund_data=None):
    factors = pd.DataFrame(index=df.index)
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    volume = pd.Series(df["Volume"].values.flatten(), index=df.index)
    high = pd.Series(df["High"].values.flatten(), index=df.index)
    low = pd.Series(df["Low"].values.flatten(), index=df.index)
    open_ = pd.Series(df["Open"].values.flatten(), index=df.index)
    daily_ret = close.pct_change()

    for days in [5, 10, 20, 60]:
        factors[f"ret_{days}d"] = close.pct_change(days)

    factors["mom_accel"] = factors["ret_5d"] - factors["ret_5d"].shift(5)
    factors["vol_20d"] = daily_ret.rolling(20).std()
    factors["vol_ratio"] = daily_ret.rolling(5).std() / daily_ret.rolling(20).std()
    factors["vol_accel"] = factors["vol_20d"].pct_change(5)
    factors["intraday_vol"] = ((high - low) / close).rolling(20).mean()
    factors["vol_spread"] = factors["intraday_vol"] - factors["vol_20d"]
    factors["vol_ma_ratio"] = volume / volume.rolling(20).mean()
    factors["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean() - 1

    obv = (daily_ret.apply(np.sign) * volume).rolling(20).sum()
    factors["obv_norm"] = obv / volume.rolling(20).sum()
    vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    factors["vwap_dev"] = close / vwap - 1
    factors["bb_pos"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() * 2)
    factors["ma_cross_s"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    factors["ma_cross_l"] = close.rolling(20).mean() / close.rolling(60).mean() - 1

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    factors["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100 - 0.5

    rolling_min = close.rolling(60).min()
    rolling_max = close.rolling(60).max()
    factors["price_pos"] = ((close - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)) - 0.5
    factors["price_range"] = (high - low) / close

    for name in ["SP500", "Nasdaq100", "EnergySector"]:
        if name in market_data:
            ref = pd.Series(market_data[name], index=market_data[f"{name}_idx"])
            ref_aligned = ref.reindex(df.index, method="ffill")
            factors[f"rel_{name}"] = close.pct_change(20) - ref_aligned.pct_change(20)
            factors[f"corr_{name}"] = daily_ret.rolling(60).corr(ref_aligned.pct_change())

    factors["ret_252d"] = close.pct_change(252)
    factors["ma200_dev"] = close / close.rolling(200).mean() - 1
    factors["sharpe_20d"] = factors["ret_20d"] / factors["vol_20d"].replace(0, np.nan)
    factors["gap"] = (open_ / close.shift(1) - 1).rolling(5).mean()

    for key in ["pe", "pb", "market_cap", "dividend_yield", "roe"]:
        value = fund_data.get(key, np.nan) if fund_data else np.nan
        factors[f"f_{key}"] = value if (not np.isnan(value) if isinstance(value, float) else True) else 0.0

    factors = factors.ffill().dropna(subset=factors.columns[:10])
    factors = factors.fillna(0.0)
    return factors
