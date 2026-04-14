import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from utils import save_fig


def plot_fold_ic(metrics_df, res_final, img_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("V4 Fold IC Analysis", fontsize=14, fontweight="bold")

    ax = axes[0]
    x = range(1, len(metrics_df) + 1)
    ax.bar([i - 0.15 for i in x], metrics_df["val_ic"], width=0.3, color="steelblue", label="Val IC", alpha=0.8)
    ax.bar([i + 0.15 for i in x], metrics_df["test_ic"], width=0.3, color="coral", label="Test IC", alpha=0.8)
    ax.axhline(y=0, color="gray", ls="--", lw=0.5)
    ax.axhline(y=0.05, color="green", ls="--", lw=0.5, label="IC=0.05")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Val vs Test IC per Fold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    daily_ic = []
    for _, group in res_final.groupby("date"):
        if len(group) < 5:
            continue
        ic, _ = stats.spearmanr(group["predicted"], group["actual"])
        if not np.isnan(ic):
            daily_ic.append(ic)
    ax.plot(np.cumsum(daily_ic), "b-", lw=1.5)
    ax.set_title(f"Cumulative IC (mean={np.mean(daily_ic):.4f})")
    ax.set_xlabel("Trading days")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(plt, img_dir, "V4_fold_ic")
    plt.close(fig)


def plot_backtest(bt_final, fold_metrics, res_plot, img_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V4 Quant Strategy (Two-Way Attention + Rolling Window)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(bt_final["date"], (1 + bt_final["gross_return"]).cumprod(), "b-", label="Gross", lw=1.5)
    ax.plot(bt_final["date"], (1 + bt_final["net_return"]).cumprod(), "r--", label="Net", lw=1.5)
    ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
    for fold_metric in fold_metrics:
        ax.axvline(x=fold_metric["test_start"], color="green", ls=":", alpha=0.4, lw=0.8)
    ax.legend()
    ax.set_title("Cumulative returns")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(bt_final["long_return"], bins=40, alpha=0.5, color="green", label="Long")
    ax.hist(bt_final["short_return"], bins=40, alpha=0.5, color="red", label="Short")
    ax.legend(fontsize=8)
    ax.set_title("Long vs Short returns")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sample = res_plot.sample(min(3000, len(res_plot)))
    ax.scatter(sample["predicted"], sample["actual"], alpha=0.1, s=5)
    ax.axhline(y=0, color="gray", ls="--")
    ax.axvline(x=0, color="gray", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Pred vs Actual")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    res_quantile = res_plot.copy()
    res_quantile["q"] = pd.qcut(res_quantile["predicted"], q=5, labels=["Q1\n(Short)", "Q2", "Q3", "Q4", "Q5\n(Long)"])
    quantile_returns = res_quantile.groupby("q", observed=False)["actual"].mean()
    bars = ax.bar(
        quantile_returns.index,
        quantile_returns.values,
        color=["#e74c3c", "#e67e22", "#95a5a6", "#27ae60", "#2ecc71"],
        edgecolor="k",
        lw=0.5,
    )
    ax.axhline(y=0, color="k", ls="--", lw=0.5)
    for bar, value in zip(bars, quantile_returns.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0002, f"{value:.4f}", ha="center", fontsize=9)
    ax.set_title("Return by quantile")
    ax.set_ylabel("Mean return")

    plt.tight_layout()
    save_fig(plt, img_dir, "V4_strategy_results")
    plt.close(fig)


def plot_daily_ic(res_plot, img_dir):
    daily_ic = []
    for dt, group in res_plot.groupby("date"):
        if len(group) < 5:
            continue
        ic, _ = stats.spearmanr(group["predicted"], group["actual"])
        if not np.isnan(ic):
            daily_ic.append({"date": dt, "ic": ic})

    ic_df = pd.DataFrame(daily_ic).sort_values("date")
    mean_ic = ic_df["ic"].mean()
    std_ic = ic_df["ic"].std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    print(f"Mean IC: {mean_ic:.4f} | Std: {std_ic:.4f} | IR: {ir:.4f} | IC>0: {(ic_df['ic'] > 0).mean():.0%}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle("V4 Daily IC Analysis", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(range(len(ic_df)), ic_df["ic"], color=["g" if x > 0 else "r" for x in ic_df["ic"]], alpha=0.6, width=1)
    ax.axhline(y=mean_ic, color="blue", ls="--", label=f"Mean={mean_ic:.4f}")
    ax.legend()
    ax.set_title("Daily IC")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ic_df["ic"].cumsum().values, "b-", lw=1.5)
    ax.set_title("Cumulative IC")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(plt, img_dir, "V4_daily_ic")
    plt.close(fig)


def plot_stock_prediction_rolling(ticker, res_plot, img_dir):
    stock_df = res_plot[res_plot["ticker"] == ticker].copy()
    if len(stock_df) == 0:
        print(f"{ticker} not found")
        return
    stock_df = stock_df.sort_values("date")
    dates = stock_df["date"].values
    actuals = stock_df["actual"].values
    preds = stock_df["predicted"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{ticker} - V4 Predictions", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(dates, actuals, "b-", label="Actual", alpha=0.8, lw=1.5)
    ax.plot(dates, preds, "r--", label="Predicted", alpha=0.8, lw=1.5)
    ax.axhline(y=0, color="gray", ls="--", lw=0.5)
    ax.legend()
    ax.set_title("Predicted vs Actual")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    deviation = preds - actuals
    ax.bar(dates, deviation, color=["green" if d >= 0 else "red" for d in deviation], alpha=0.6, width=2)
    ax.axhline(y=deviation.mean(), color="blue", ls="--", label=f"Mean:{deviation.mean():.4f}")
    ax.legend()
    ax.set_title("Deviation")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    correct = (np.sign(preds) == np.sign(actuals)).astype(int)
    ax.plot(dates, pd.Series(correct).expanding().mean(), "purple", lw=2)
    ax.axhline(y=0.5, color="gray", ls="--")
    ax.set_ylim(0.3, 0.8)
    ax.set_title(f"Direction accuracy ({correct.mean():.1%})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(plt, img_dir, f"{ticker}_V4")
    plt.close(fig)

    corr = np.corrcoef(preds, actuals)[0, 1]
    print(f"{ticker}: {len(stock_df)} samples | Dir acc: {correct.mean():.1%} | Corr: {corr:.4f}")
