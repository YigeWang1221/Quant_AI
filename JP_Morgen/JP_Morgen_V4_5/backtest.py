import pandas as pd
import numpy as np


def backtest_from_predictions(res, top_n=3, tx_cost=0.001, rebal_freq=5, rev=False):
    res = res.copy()
    realized_col = "raw_actual" if "raw_actual" in res.columns else "actual"
    if rev:
        res["predicted"] = -res["predicted"]
    dg = res.groupby("date")
    rd = sorted(res["date"].unique())[::rebal_freq]
    rows, pl, ps = [], set(), set()
    for dt in rd:
        if dt not in dg.groups:
            continue
        g = dg.get_group(dt)
        if len(g) < 2 * top_n:
            continue
        g = g.sort_values("predicted", ascending=False)
        lt = set(g.head(top_n)["ticker"])
        st = set(g.tail(top_n)["ticker"])
        lr = g.head(top_n)[realized_col].mean()
        sr = -g.tail(top_n)[realized_col].mean()
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
