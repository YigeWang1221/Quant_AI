from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import END_DATE, LOOKBACK, MIN_HISTORY_YEARS, MIN_STOCKS_PER_DAY, MIN_STOCKS_RATIO, SCALER_WINDOW, START_DATE
from data_loader import get_or_load_data
from features import compute_v3_factors


class DailyDataset(Dataset):
    def __init__(self, dates, daily_samples, max_stocks=None):
        self.dates = dates
        self.daily_samples = daily_samples
        self.max_stocks = max_stocks or max(len(daily_samples[dt]["tickers"]) for dt in dates)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        dt = self.dates[idx]
        sample = self.daily_samples[dt]
        x = torch.FloatTensor(sample["X"])
        y = torch.FloatTensor(sample["y"])
        n_stocks = x.shape[0]
        if n_stocks < self.max_stocks:
            x = torch.cat([x, torch.zeros(self.max_stocks - n_stocks, x.shape[1], x.shape[2])])
            y = torch.cat([y, torch.zeros(self.max_stocks - n_stocks)])
        mask = torch.zeros(self.max_stocks)
        mask[:n_stocks] = 1.0
        return x, y, mask


def process_and_normalize_data(start=START_DATE, end=END_DATE):
    raw_data, market_data, fundamentals = get_or_load_data(start=start, end=end)

    sample_ticker = list(raw_data.keys())[0]
    sample_factors = compute_v3_factors(raw_data[sample_ticker], market_data, fundamentals.get(sample_ticker, {}))
    print(f"Factors: {sample_factors.shape[1]}")

    all_stock = {}
    for ticker, df in raw_data.items():
        factors = compute_v3_factors(df, market_data, fundamentals.get(ticker, {}))
        close_series = pd.Series(df["Close"].values.flatten(), index=df.index)
        target = close_series.pct_change(5).shift(-5)
        idx = factors.index.intersection(target.dropna().index)
        if len(idx) == 0:
            continue
        all_stock[ticker] = {
            "factors": factors.loc[idx],
            "target": target.loc[idx],
            "raw_target": target.loc[idx],
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
    }
    print(f"full_data ready | {len(valid_dates)} days | {num_factors} factors")
    return full_data
