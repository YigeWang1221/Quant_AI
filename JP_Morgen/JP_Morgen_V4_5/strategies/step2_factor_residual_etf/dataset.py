from collections import Counter

import numpy as np
import pandas as pd

from config import END_DATE, LOOKBACK, MARKET_TICKERS, MIN_HISTORY_YEARS, MIN_STOCKS_PER_DAY, MIN_STOCKS_RATIO, SCALER_WINDOW, START_DATE
from data_loader import get_or_load_data
from features import compute_v3_factors


FORWARD_HORIZON = 5
TARGET_STRATEGY = "factor_residual_etf"
DEFAULT_BETA_WINDOW = 120
DEFAULT_BETA_MIN_OBS = 60
ETF_PROXY_TICKERS = ("SPY", "QQQ", "XLE", "TLT", "GLD")


def _build_etf_return_frames(market_data, horizon=FORWARD_HORIZON):
    etf_close = {}
    available_proxies = []
    for ticker in ETF_PROXY_TICKERS:
        market_name = MARKET_TICKERS.get(ticker)
        idx_key = f"{market_name}_idx"
        if market_name in market_data and idx_key in market_data:
            etf_close[ticker] = pd.Series(market_data[market_name], index=pd.DatetimeIndex(market_data[idx_key])).sort_index()
            available_proxies.append(ticker)

    if not etf_close:
        raise ValueError("ETF residual strategy requires cached ETF proxy series in market_data.")

    close_df = pd.DataFrame(etf_close).sort_index()
    daily_returns = close_df.pct_change()
    forward_returns = close_df.pct_change(horizon).shift(-horizon)
    return daily_returns, forward_returns, available_proxies


def _estimate_factor_betas(stock_window, factor_window, min_obs):
    if factor_window.ndim == 1:
        factor_window = factor_window.reshape(-1, 1)
    if factor_window.shape[1] == 0:
        return None

    valid_mask = np.isfinite(stock_window) & np.all(np.isfinite(factor_window), axis=1)
    if valid_mask.sum() < min_obs:
        return None

    y = stock_window[valid_mask]
    x = factor_window[valid_mask]
    design = np.column_stack([np.ones(len(y)), x])

    try:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coeffs[1:]


def build_factor_residual_targets(
    raw_data,
    market_data,
    horizon=FORWARD_HORIZON,
    beta_window=DEFAULT_BETA_WINDOW,
    beta_min_obs=DEFAULT_BETA_MIN_OBS,
):
    etf_daily_returns, etf_forward_returns, available_proxies = _build_etf_return_frames(market_data, horizon=horizon)

    factor_residual_targets = {}
    raw_forward_targets = {}
    total_target_days = 0
    residualized_days = 0
    raw_fallback_days = 0

    forward_factor_cols = [f"{ticker}_fwd" for ticker in available_proxies]
    for ticker, df in raw_data.items():
        close_series = pd.Series(df["Close"].values.flatten(), index=df.index).sort_index()
        stock_daily_returns = close_series.pct_change()
        stock_forward_returns = close_series.pct_change(horizon).shift(-horizon)

        combined = pd.DataFrame(
            {
                "stock_daily": stock_daily_returns,
                "stock_forward": stock_forward_returns,
            }
        )
        combined = combined.join(etf_daily_returns[available_proxies], how="left")
        combined = combined.join(etf_forward_returns[available_proxies].add_suffix("_fwd"), how="left")
        combined = combined.sort_index()

        stock_daily_arr = combined["stock_daily"].to_numpy(dtype=float)
        stock_forward_arr = combined["stock_forward"].to_numpy(dtype=float)
        factor_daily_arr = combined[available_proxies].to_numpy(dtype=float)
        factor_forward_arr = combined[forward_factor_cols].to_numpy(dtype=float)

        residual_values = np.full(len(combined), np.nan, dtype=float)
        for i in range(len(combined)):
            raw_target = stock_forward_arr[i]
            if not np.isfinite(raw_target):
                continue

            total_target_days += 1
            future_factor_returns = factor_forward_arr[i]
            active_factor_mask = np.isfinite(future_factor_returns)

            if not active_factor_mask.any():
                residual_values[i] = raw_target
                raw_fallback_days += 1
                continue

            start = max(0, i - beta_window + 1)
            betas = _estimate_factor_betas(
                stock_daily_arr[start:i + 1],
                factor_daily_arr[start:i + 1][:, active_factor_mask],
                beta_min_obs,
            )
            if betas is None:
                residual_values[i] = raw_target
                raw_fallback_days += 1
                continue

            residual_values[i] = raw_target - float(np.dot(betas, future_factor_returns[active_factor_mask]))
            residualized_days += 1

        factor_residual_targets[ticker] = pd.Series(residual_values, index=combined.index, name=ticker)
        raw_forward_targets[ticker] = pd.Series(stock_forward_arr, index=combined.index, name=ticker)

    residual_ratio = residualized_days / max(total_target_days, 1)
    print(f"Targets: ETF-residual {horizon}-day forward returns | proxies: {', '.join(available_proxies)}")
    print(f"Beta estimation: trailing {beta_window} daily returns | min obs {beta_min_obs}")
    print(f"Residualized labels: {residualized_days}/{total_target_days} ({residual_ratio:.1%}) | raw fallback: {raw_fallback_days}")
    return factor_residual_targets, raw_forward_targets, available_proxies


def process_and_normalize_data(
    start=START_DATE,
    end=END_DATE,
    beta_window=DEFAULT_BETA_WINDOW,
    beta_min_obs=DEFAULT_BETA_MIN_OBS,
):
    raw_data, market_data, fundamentals = get_or_load_data(start=start, end=end)

    sample_ticker = list(raw_data.keys())[0]
    sample_factors = compute_v3_factors(raw_data[sample_ticker], market_data, fundamentals.get(sample_ticker, {}))
    print("Strategy: Step2 FactorResidualETF")
    print(f"Factors: {sample_factors.shape[1]}")

    target_by_ticker, raw_target_by_ticker, available_proxies = build_factor_residual_targets(
        raw_data,
        market_data,
        horizon=FORWARD_HORIZON,
        beta_window=beta_window,
        beta_min_obs=beta_min_obs,
    )

    all_stock = {}
    for ticker, df in raw_data.items():
        factors = compute_v3_factors(df, market_data, fundamentals.get(ticker, {}))
        target = target_by_ticker.get(ticker)
        raw_target = raw_target_by_ticker.get(ticker)
        if target is None:
            continue
        idx = factors.index.intersection(target.dropna().index)
        if len(idx) == 0:
            continue
        all_stock[ticker] = {
            "factors": factors.loc[idx],
            "target": target.loc[idx],
            "raw_target": raw_target.loc[idx],
        }

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=MIN_HISTORY_YEARS)
    short_history = [ticker for ticker in all_stock if all_stock[ticker]["factors"].index.min() > cutoff]
    if short_history:
        print(f"Removing {len(short_history)} stocks with < {MIN_HISTORY_YEARS} years")
        for ticker in short_history:
            del all_stock[ticker]

    num_factors = list(all_stock.values())[0]["factors"].shape[1]
    print(f"Keeping {len(all_stock)} stocks | {num_factors} factors")

    date_counts = Counter()
    for stock_blob in all_stock.values():
        for dt in stock_blob["factors"].index:
            date_counts[dt] += 1

    n_stocks = len(all_stock)
    min_per_day = int(n_stocks * MIN_STOCKS_RATIO)
    all_dates = sorted([dt for dt, count in date_counts.items() if count >= min_per_day])
    print(f"Valid days: {len(all_dates)} | {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")

    all_dates_set = set(all_dates)
    stock_data = {}
    for ticker, stock_blob in all_stock.items():
        factors_df = stock_blob["factors"]
        target = stock_blob["target"]
        raw_target = stock_blob["raw_target"]
        factor_values = factors_df.values
        dates_idx = factors_df.index
        normalized_rows = {}

        for i in range(max(SCALER_WINDOW, LOOKBACK), len(factor_values)):
            window = factor_values[max(0, i - SCALER_WINDOW):i]
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0)
            std[std < 1e-8] = 1.0
            scaled = (factor_values[i - LOOKBACK:i] - mean) / std
            if scaled.shape != (LOOKBACK, num_factors) or np.any(np.isnan(scaled)):
                continue
            dt = dates_idx[i]
            if dt in all_dates_set and not np.isnan(target.iloc[i]) and not np.isnan(raw_target.iloc[i]):
                normalized_rows[dt] = {
                    "x": scaled.astype(np.float32),
                    "y": target.iloc[i],
                    "raw_y": raw_target.iloc[i],
                }

        if normalized_rows:
            stock_data[ticker] = normalized_rows

    tickers_list = sorted(stock_data.keys())
    dropped = [ticker for ticker in all_stock if ticker not in stock_data]
    print(f"Dropped {len(dropped)} stocks in normalization")
    if dropped:
        ticker = dropped[0]
        factors_df = all_stock[ticker]["factors"]
        print(f"  Example: {ticker} has {len(factors_df)} rows, min_i = {max(SCALER_WINDOW, LOOKBACK)}")
        factor_values = factors_df.values
        nan_count = 0
        for i in range(max(SCALER_WINDOW, LOOKBACK), len(factor_values)):
            window = factor_values[max(0, i - SCALER_WINDOW):i]
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0)
            std[std < 1e-8] = 1.0
            scaled = (factor_values[i - LOOKBACK:i] - mean) / std
            if np.any(np.isnan(scaled)):
                nan_count += 1
        print(f"  Rows with NaN after norm: {nan_count}/{len(factor_values) - max(SCALER_WINDOW, LOOKBACK)}")
        print(f"  NaN per column in raw factors: {np.isnan(factor_values).sum(axis=0).tolist()}")
    print(f"Stocks with data: {len(tickers_list)}")

    daily_samples = {}
    for dt in all_dates:
        day_x, day_y, day_raw_y, day_tickers = [], [], [], []
        for ticker in tickers_list:
            if dt in stock_data[ticker]:
                day_x.append(stock_data[ticker][dt]["x"])
                day_y.append(stock_data[ticker][dt]["y"])
                day_raw_y.append(stock_data[ticker][dt]["raw_y"])
                day_tickers.append(ticker)
        if len(day_x) >= MIN_STOCKS_PER_DAY:
            daily_samples[dt] = {
                "X": np.stack(day_x),
                "y": np.array(day_y, dtype=np.float32),
                "raw_y": np.array(day_raw_y, dtype=np.float32),
                "tickers": day_tickers,
            }

    valid_dates = sorted(daily_samples.keys())
    n_per_day = [len(daily_samples[dt]["tickers"]) for dt in valid_dates]
    print(f"Daily dataset: {len(valid_dates)} days | stocks/day: {min(n_per_day)}-{max(n_per_day)} (avg {np.mean(n_per_day):.0f})")

    full_data = {
        "daily_samples": daily_samples,
        "valid_dates": valid_dates,
        "tickers_list": tickers_list,
        "num_factors": num_factors,
        "target_strategy": TARGET_STRATEGY,
        "beta_window": beta_window,
        "beta_min_obs": beta_min_obs,
        "etf_proxy_tickers": list(available_proxies),
    }
    print(f"full_data ready | {len(valid_dates)} days | {num_factors} factors")
    return full_data
