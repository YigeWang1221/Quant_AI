import json
import os
import pickle

import numpy as np
import pandas as pd

from config import END_DATE, MARKET_TICKERS, META_FILE, SAVE_DIR, START_DATE, STOCK_UNIVERSE_FLAT


def save_meta(start, end, num_stocks):
    with open(META_FILE, "w") as file:
        json.dump(
            {
                "start_date": start,
                "end_date": end,
                "num_stocks": num_stocks,
                "saved_at": pd.Timestamp.now().isoformat(),
            },
            file,
            indent=2,
        )


def check_date_range(start, end):
    if not os.path.exists(META_FILE):
        return False, "no meta.json"
    with open(META_FILE, "r") as file:
        meta = json.load(file)
    if meta.get("start_date", "") > start:
        return False, f"local start {meta['start_date']} > {start}"
    if meta.get("end_date", "") < end:
        return False, f"local end {meta['end_date']} < {end}"
    print(f"OK: local [{meta['start_date']} ~ {meta['end_date']}] covers [{start} ~ {end}]")
    return True, "OK"


def download_all(start=START_DATE, end=END_DATE):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("`yfinance` is required when local cached data is unavailable.") from exc

    print(f"Downloading [{start} ~ {end}]...")
    raw = {}
    for ticker in STOCK_UNIVERSE_FLAT:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 200:
                raw[ticker] = df
                print(f"  + {ticker}: {len(df)}")
        except Exception:
            print(f"  x {ticker}")

    market = {}
    for ticker, name in MARKET_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 200:
                market[name] = df["Close"].values.flatten()
                market[f"{name}_idx"] = df.index
        except Exception:
            pass

    fundamentals = {}
    for ticker in raw:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                "pe": info.get("trailingPE", np.nan),
                "pb": info.get("priceToBook", np.nan),
                "market_cap": info.get("marketCap", np.nan),
                "dividend_yield": info.get("dividendYield", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
            }
        except Exception:
            fundamentals[ticker] = {}
    return raw, market, fundamentals


def save_data(raw, market, fundamentals, start=START_DATE, end=END_DATE):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for ticker, df in raw.items():
        df_save = df.copy()
        df_save.columns = [str(col).split(",")[0].strip().strip("('\"") for col in df_save.columns]
        df_save.to_csv(os.path.join(SAVE_DIR, f"{ticker}.csv"))
    with open(os.path.join(SAVE_DIR, "market.pkl"), "wb") as file:
        pickle.dump(market, file)
    with open(os.path.join(SAVE_DIR, "fund.pkl"), "wb") as file:
        pickle.dump(fundamentals, file)
    save_meta(start, end, len(raw))
    print(f"Saved {len(raw)} stocks to {SAVE_DIR}/")


def load_data(start=START_DATE, end=END_DATE):
    raw = {}
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for file_name in os.listdir(SAVE_DIR):
        if not file_name.endswith(".csv"):
            continue
        ticker = file_name.replace(".csv", "")
        df = pd.read_csv(os.path.join(SAVE_DIR, file_name))
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif df.columns[0] in ("Unnamed: 0", "date"):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
        else:
            df.index = pd.to_datetime(df.index)
        df.columns = [str(col).split(",")[0].strip().strip("('\"") for col in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(df) > 200:
            raw[ticker] = df

    with open(os.path.join(SAVE_DIR, "market.pkl"), "rb") as file:
        market = pickle.load(file)
    with open(os.path.join(SAVE_DIR, "fund.pkl"), "rb") as file:
        fundamentals = pickle.load(file)

    for name in ["SP500", "Nasdaq100", "EnergySector", "Bond20Y", "Gold"]:
        idx_key = f"{name}_idx"
        if name in market and idx_key in market:
            idx = market[idx_key]
            mask = (idx >= start_ts) & (idx <= end_ts)
            market[name] = market[name][mask]
            market[idx_key] = idx[mask]

    print(f"Loaded {len(raw)} stocks [{start} ~ {end}]")
    return raw, market, fundamentals


def get_or_load_data(start=START_DATE, end=END_DATE):
    data_valid = False
    market_path = os.path.join(SAVE_DIR, "market.pkl")
    if os.path.exists(market_path):
        data_valid, reason = check_date_range(start, end)
        if not data_valid:
            print(f"Re-downloading: {reason}")
    if data_valid:
        return load_data(start=start, end=end)
    raw_data, market_data, fundamentals = download_all(start=start, end=end)
    save_data(raw_data, market_data, fundamentals, start=start, end=end)
    return raw_data, market_data, fundamentals
