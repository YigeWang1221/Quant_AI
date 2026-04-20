import pandas as pd
import numpy as np


def prepare_backtest_predictions(res, score_smooth_window=1, score_smooth_method="off"):
    res = res.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    res["signal_raw"] = res["predicted"]

    method = str(score_smooth_method).lower()
    if score_smooth_window > 1 and method != "off":
        grouped = res.groupby("ticker", group_keys=False)["predicted"]
        if method == "sma":
            smoothed = grouped.transform(lambda s: s.rolling(score_smooth_window, min_periods=1).mean())
        elif method == "ewm":
            smoothed = grouped.transform(lambda s: s.ewm(span=score_smooth_window, adjust=False, min_periods=1).mean())
        else:
            raise ValueError(f"Unsupported score_smooth_method: {score_smooth_method}")
    else:
        smoothed = res["predicted"]

    res["trade_signal_raw"] = smoothed
    date_mean = res.groupby("date")["trade_signal_raw"].transform("mean")
    date_std = res.groupby("date")["trade_signal_raw"].transform("std").replace(0, np.nan)
    res["trade_signal"] = ((res["trade_signal_raw"] - date_mean) / date_std).fillna(0.0)
    return res.sort_values("date").reset_index(drop=True)


def _select_positions(g, signal_col, top_n, prev_longs, prev_shorts, no_trade_band):
    working = g.copy()
    working["prev_long_bonus"] = working["ticker"].isin(prev_longs).astype(float) * no_trade_band
    working["long_priority"] = working[signal_col] + working["prev_long_bonus"]
    long_ranked = working.sort_values(["long_priority", signal_col], ascending=[False, False])
    long_names = list(long_ranked.head(top_n)["ticker"])

    remaining = working[~working["ticker"].isin(long_names)].copy()
    remaining["prev_short_bonus"] = remaining["ticker"].isin(prev_shorts).astype(float) * no_trade_band
    remaining["short_priority"] = -remaining[signal_col] + remaining["prev_short_bonus"]
    short_ranked = remaining.sort_values(["short_priority", signal_col], ascending=[False, True])
    short_names = list(short_ranked.head(top_n)["ticker"])

    return long_names, short_names


def backtest_from_predictions(
    res,
    top_n=3,
    tx_cost=0.001,
    rebal_freq=5,
    rev=False,
    score_smooth_window=1,
    score_smooth_method="off",
    no_trade_band=0.0,
):
    res = res.copy()
    if "trade_signal" not in res.columns:
        res = prepare_backtest_predictions(
            res,
            score_smooth_window=score_smooth_window,
            score_smooth_method=score_smooth_method,
        )
    realized_col = "raw_actual" if "raw_actual" in res.columns else "actual"
    signal_col = "trade_signal"
    if rev:
        res[signal_col] = -res[signal_col]
    dg = res.groupby("date")
    rd = sorted(res["date"].unique())[::rebal_freq]
    rows, pl, ps = [], set(), set()
    for dt in rd:
        if dt not in dg.groups:
            continue
        g = dg.get_group(dt).copy()
        if len(g) < 2 * top_n:
            continue
        long_names, short_names = _select_positions(
            g,
            signal_col=signal_col,
            top_n=top_n,
            prev_longs=pl,
            prev_shorts=ps,
            no_trade_band=no_trade_band,
        )
        lt = set(long_names)
        st = set(short_names)
        long_slice = g[g["ticker"].isin(long_names)]
        short_slice = g[g["ticker"].isin(short_names)]
        lr = long_slice[realized_col].mean()
        sr = -short_slice[realized_col].mean()
        to = (len(lt - pl) + len(st - ps)) / (2 * top_n)
        cost = to * tx_cost * 2
        gross = (lr + sr) / 2
        rows.append(
            {
                "date": dt,
                "long_return": lr,
                "short_return": sr,
                "gross_return": gross,
                "net_return": gross - cost,
                "turnover": to,
                "cost": cost,
                "return_source": realized_col,
                "avg_long_signal": long_slice[signal_col].mean(),
                "avg_short_signal": short_slice[signal_col].mean(),
                "top_n": top_n,
                "rebal_freq": rebal_freq,
                "score_smooth_window": score_smooth_window,
                "score_smooth_method": score_smooth_method,
                "no_trade_band": no_trade_band,
            }
        )
        pl, ps = lt, st
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def evaluate(bt, rebal_freq=5):
    ppy = 252 / rebal_freq
    for lab, col in [("Gross", "gross_return"), ("Net", "net_return")]:
        r = bt[col]
        cum = (1 + r).cumprod()
        tot = cum.iloc[-1] - 1
        n = len(r)
        ar = (1 + tot) ** (ppy / n) - 1
        av = r.std() * np.sqrt(ppy)
        sh = (ar - 0.04) / av if av > 0 else 0
        md = ((cum - cum.cummax()) / cum.cummax()).min()
        wr = (r > 0).mean()
        print(f'\n  {"="*45}\n  {lab}\n  {"="*45}')
        print(f'    Total:  {tot:.2%}\n    Annual: {ar:.2%}\n    Vol:    {av:.2%}\n    Sharpe: {sh:.2f}\n    MaxDD:  {md:.2%}\n    WinRate:{wr:.0%}')

    print(f'\n  {"="*45}\n  By Year\n  {"="*45}')
    bc = bt.copy()
    bc["year"] = pd.to_datetime(bc["date"]).dt.year
    for year, grp in bc.groupby("year"):
        yr = grp["net_return"]
        yr_tot = (1 + yr).cumprod().iloc[-1] - 1
        yr_sh = (yr.mean() / yr.std() * np.sqrt(ppy)) if yr.std() > 0 else 0
        print(f'    {year}: Ret={yr_tot:.2%} Sharpe={yr_sh:.2f} N={len(yr)}')
