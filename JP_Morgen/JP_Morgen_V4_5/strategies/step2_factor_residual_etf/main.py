import argparse
import os
import sys
from pathlib import Path

import pandas as pd


STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STRATEGY_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backtest import backtest_from_predictions, evaluate, prepare_backtest_predictions
from config import (
    DEFAULT_AMP_DTYPE,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_LISTNET_WEIGHT,
    DEFAULT_LR,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TEMPERATURE,
)
from utils import create_run_dirs, get_device, setup_logging
from visualization import plot_backtest, plot_daily_ic, plot_fold_ic, plot_stock_prediction_rolling

from strategies.step2_factor_residual_etf.dataset import (
    DEFAULT_BETA_MIN_OBS,
    DEFAULT_BETA_WINDOW,
    ETF_PROXY_TICKERS,
    FORWARD_HORIZON,
    TARGET_STRATEGY,
    process_and_normalize_data,
)
from strategies.step2_factor_residual_etf.trainer import describe_folds, generate_folds, train_one_fold


STRATEGY_CODE = "step2_factor_residual_etf"
STRATEGY_NAME = "Step2 FactorResidualETF"
STRATEGY_SUMMARY = (
    "Use trailing ETF betas to strip out predicted systematic return from each stock's "
    "5-day forward return so the model trains on a cleaner idiosyncratic target."
)
TARGET_DESCRIPTION = (
    f"Label = stock {FORWARD_HORIZON}-day forward return minus trailing-beta ETF-implied "
    f"{FORWARD_HORIZON}-day return using {', '.join(ETF_PROXY_TICKERS)}."
)
TRAINER_DESCRIPTION = (
    "Notebook-aligned local-memory trainer: precomputed day tensors stay on CPU, "
    "batched forward passes run on GPU, and each batch moves to device only when needed to reduce OOM risk."
)
REFERENCE_NOTEBOOK = "JP_Morgen_V4_MPS.ipynb"
LOG_EXPERIMENT_NAME = "step2-factor-residual-etf"
STEP2_DEFAULT_BATCH_DAYS = 20
STEP2_DEFAULT_NUM_EPOCHS = 100
STEP2_DEFAULT_PATIENCE = 15
STEP2_DEFAULT_AMP_MODE = "on"
STEP2_DEFAULT_STOCK_PLOT = "AAPL"
STEP2_DEFAULT_TOP_N = 5
STEP2_DEFAULT_REBAL_FREQ = 5
STEP2_DEFAULT_SCORE_SMOOTH_WINDOW = 3
STEP2_DEFAULT_SCORE_SMOOTH_METHOD = "ewm"
STEP2_DEFAULT_NO_TRADE_BAND = 0.30


def parse_args():
    parser = argparse.ArgumentParser(description="JP Morgan Quant V4.5 Step2 ETF Residual Training")
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL, help="Transformer hidden dim")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of Two-Way blocks")
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--listnet_weight", type=float, default=DEFAULT_LISTNET_WEIGHT, help="Weight of ListNet Loss")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="ListNet temperature")
    parser.add_argument("--num_epochs", type=int, default=STEP2_DEFAULT_NUM_EPOCHS, help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=STEP2_DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--batch_days", type=int, default=STEP2_DEFAULT_BATCH_DAYS, help="Days packaged into a batch")
    parser.add_argument("--top_n", type=int, default=STEP2_DEFAULT_TOP_N, help="Top N stocks")
    parser.add_argument("--rebal_freq", type=int, default=STEP2_DEFAULT_REBAL_FREQ, help="Rebalance frequency")
    parser.add_argument("--stock_plot", type=str, default=STEP2_DEFAULT_STOCK_PLOT, help="Ticker to visualize")
    parser.add_argument("--beta_window", type=int, default=DEFAULT_BETA_WINDOW, help="Trailing window for ETF beta estimation")
    parser.add_argument("--beta_min_obs", type=int, default=DEFAULT_BETA_MIN_OBS, help="Minimum valid daily returns to fit ETF betas")
    parser.add_argument(
        "--score_smooth_window",
        type=int,
        default=STEP2_DEFAULT_SCORE_SMOOTH_WINDOW,
        help="Number of prediction observations used to smooth each ticker's trade score",
    )
    parser.add_argument(
        "--score_smooth_method",
        type=str,
        default=STEP2_DEFAULT_SCORE_SMOOTH_METHOD,
        choices=["off", "sma", "ewm"],
        help="Smoothing method for trade scores before portfolio construction",
    )
    parser.add_argument(
        "--no_trade_band",
        type=float,
        default=STEP2_DEFAULT_NO_TRADE_BAND,
        help="Incumbency bonus in z-score units before a position is replaced at rebalance",
    )
    parser.add_argument(
        "--amp_mode",
        type=str,
        default=STEP2_DEFAULT_AMP_MODE,
        choices=["auto", "on", "off"],
        help="Use CUDA mixed precision in the forward pass",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=DEFAULT_AMP_DTYPE,
        choices=["float16", "bfloat16"],
        help="Mixed precision dtype for CUDA forward pass",
    )
    return parser.parse_args()


def build_run_label(args):
    return (
        f"target-{TARGET_STRATEGY}"
        f"__trainer-cpu-batch"
        f"__bw{args.beta_window}-mo{args.beta_min_obs}"
        f"__b{args.batch_days}"
        f"__d{args.d_model}-l{args.num_layers}-h{args.nhead}"
        f"__lr{args.lr:g}"
        f"__ep{args.num_epochs}-p{args.patience}"
        f"__tn{args.top_n}-rf{args.rebal_freq}"
        f"__sm-{args.score_smooth_method}{args.score_smooth_window}"
        f"__ntb{args.no_trade_band:g}"
        f"__amp-{args.amp_mode}"
    )


def build_parameter_manifest(args):
    return {
        "d_model": {"value": args.d_model, "description": "Transformer hidden dimension."},
        "num_layers": {"value": args.num_layers, "description": "Number of Two-Way attention blocks."},
        "nhead": {"value": args.nhead, "description": "Attention heads per transformer layer."},
        "dropout": {"value": args.dropout, "description": "Dropout applied in projection and head layers."},
        "listnet_weight": {
            "value": args.listnet_weight,
            "description": "Weight for ListNet ranking loss; step2 keeps this at 0.0 to isolate the target change first.",
        },
        "temperature": {"value": args.temperature, "description": "ListNet softmax temperature."},
        "num_epochs": {"value": args.num_epochs, "description": "Maximum epochs per fold."},
        "patience": {"value": args.patience, "description": "Early stopping patience after epoch 20."},
        "lr": {"value": args.lr, "description": "AdamW learning rate."},
        "batch_days": {
            "value": args.batch_days,
            "description": "Number of trading days grouped into one training batch before GPU transfer.",
        },
        "top_n": {"value": args.top_n, "description": "Top and bottom names used in long-short backtest."},
        "rebal_freq": {"value": args.rebal_freq, "description": "Backtest rebalance interval in trading days."},
        "score_smooth_window": {
            "value": args.score_smooth_window,
            "description": "Number of per-ticker predictions used for score smoothing before trading.",
        },
        "score_smooth_method": {
            "value": args.score_smooth_method,
            "description": "Trade-score smoothing method used before rebalance selection.",
        },
        "no_trade_band": {
            "value": args.no_trade_band,
            "description": "Incumbency bonus in normalized signal units before replacing an existing position.",
        },
        "stock_plot": {"value": args.stock_plot, "description": "Ticker used for rolling prediction visualization."},
        "beta_window": {"value": args.beta_window, "description": "Trailing daily-return window used to estimate ETF betas."},
        "beta_min_obs": {"value": args.beta_min_obs, "description": "Minimum valid daily-return observations required to fit ETF betas."},
        "amp_mode": {
            "value": args.amp_mode,
            "description": "CUDA mixed precision forward mode. Step2 now defaults to on for faster local training.",
        },
        "amp_dtype": {"value": args.amp_dtype, "description": "Mixed precision dtype when AMP is enabled."},
    }


def build_run_manifest(args, device, run_paths):
    parameter_manifest = build_parameter_manifest(args)
    return {
        "strategy": {
            "code": STRATEGY_CODE,
            "name": STRATEGY_NAME,
            "summary": STRATEGY_SUMMARY,
            "target_strategy": TARGET_STRATEGY,
            "target_description": TARGET_DESCRIPTION,
            "trainer_description": TRAINER_DESCRIPTION,
            "reference_notebook": REFERENCE_NOTEBOOK,
            "entrypoint": str((STRATEGY_DIR / "main.py").resolve()),
            "strategy_dir": str(STRATEGY_DIR.resolve()),
            "etf_proxy_tickers": list(ETF_PROXY_TICKERS),
        },
        "runtime": {
            "device": str(device),
            "amp_enabled": bool(getattr(args, "amp_enabled", False)),
            "amp_dtype": args.amp_dtype,
        },
        "artifacts": {
            "run_name": run_paths["run_name"],
            "run_dir": run_paths["run_dir"],
            "stdout_log": run_paths["log_file"],
            "stderr_log": run_paths["err_file"],
            "manifest_json": run_paths["manifest_json"],
            "manifest_txt": run_paths["manifest_txt"],
        },
        "parameters": {name: payload["value"] for name, payload in parameter_manifest.items()},
        "parameter_descriptions": {name: payload["description"] for name, payload in parameter_manifest.items()},
    }


def build_log_header_lines(args, device, run_paths):
    parameter_manifest = build_parameter_manifest(args)
    lines = [
        "Experiment Manifest",
        f"  Strategy code: {STRATEGY_CODE}",
        f"  Strategy name: {STRATEGY_NAME}",
        f"  Strategy summary: {STRATEGY_SUMMARY}",
        f"  Target label: {TARGET_DESCRIPTION}",
        f"  Trainer profile: {TRAINER_DESCRIPTION}",
        f"  ETF proxies: {', '.join(ETF_PROXY_TICKERS)}",
        f"  Reference notebook: {REFERENCE_NOTEBOOK}",
        f"  Strategy dir: {STRATEGY_DIR}",
        f"  Device: {device}",
        f"  AMP active: {getattr(args, 'amp_enabled', False)} ({args.amp_dtype})",
        "",
        "Loaded Parameters",
    ]
    for name, payload in parameter_manifest.items():
        lines.append(f"  {name}={payload['value']} | {payload['description']}")
    lines.extend(
        [
            "",
            "Artifact Files",
            f"  stdout: {run_paths['log_file']}",
            f"  stderr: {run_paths['err_file']}",
            f"  manifest_json: {run_paths['manifest_json']}",
            f"  manifest_txt: {run_paths['manifest_txt']}",
            "",
        ]
    )
    return lines


def main():
    args = parse_args()

    device = get_device()
    args.amp_enabled = device.type == "cuda" and args.amp_mode != "off"
    run_label = build_run_label(args)
    run_paths = create_run_dirs(run_label=run_label, experiment_name=LOG_EXPERIMENT_NAME)
    run_manifest = build_run_manifest(args, device, run_paths)
    log_header_lines = build_log_header_lines(args, device, run_paths)
    setup_logging(run_paths, header_lines=log_header_lines, manifest=run_manifest)
    print(f"Strategy variant: {STRATEGY_NAME}")
    print(f"Hyperparameters: {vars(args)}")
    if args.amp_mode == "on" and device.type != "cuda":
        print("AMP requested, but CUDA is unavailable. Falling back to full precision.")
    print(f"Forward AMP: {'on' if args.amp_enabled else 'off'} ({args.amp_dtype}) | Loss precision: fp32")

    full_data = process_and_normalize_data(beta_window=args.beta_window, beta_min_obs=args.beta_min_obs)
    print(f"Active ETF proxies in data: {', '.join(full_data['etf_proxy_tickers'])}")
    valid_dates_dt = [pd.Timestamp(d).to_pydatetime() for d in full_data["valid_dates"]]
    date_range_years = (max(valid_dates_dt) - min(valid_dates_dt)).days / 365.25

    print(f"Data spans {date_range_years:.1f} years")
    if date_range_years >= 7:
        mt, vm, tm = 3, 6, 6
    elif date_range_years >= 4:
        mt, vm, tm = 2, 4, 4
    else:
        mt, vm, tm = 1, 3, 3

    folds = generate_folds(valid_dates_dt, val_months=vm, test_months=tm, min_train_years=mt)
    describe_folds(folds)

    all_results = []
    fold_metrics = []
    for i, fold in enumerate(folds):
        print(f"\n{'=' * 60}\nFold {i + 1}/{len(folds)}: test [{fold['test_start'].strftime('%Y-%m-%d')} ~ {fold['test_end'].strftime('%Y-%m-%d')}]\n{'=' * 60}")
        result, val_ic, test_ic = train_one_fold(fold, full_data, device, args)
        if len(result) > 0:
            all_results.append(result)
            fold_metrics.append(
                {
                    "fold": i + 1,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "val_ic": val_ic,
                    "test_ic": test_ic,
                    "n_samples": len(result),
                }
            )

    if not all_results:
        print("No results generated.")
        return

    res_final = pd.concat(all_results, ignore_index=True).sort_values("date").reset_index(drop=True)
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{'=' * 60}\nV4 Rolling Window Summary\n{'=' * 60}")
    for _, row in metrics_df.iterrows():
        print(f"  Fold {row['fold']}: Val IC={row['val_ic']:.4f} | Test IC={row['test_ic']:.4f} | [{row['test_start'].strftime('%Y-%m')} ~ {row['test_end'].strftime('%Y-%m')}]")
    print(f"\n  Avg Val IC:  {metrics_df['val_ic'].mean():.4f}")
    print(f"  Avg Test IC: {metrics_df['test_ic'].mean():.4f}")
    print(f"  IC gap: {metrics_df['val_ic'].mean() - metrics_df['test_ic'].mean():.4f}")
    print(f"  Total predictions: {len(res_final):,}")

    signal_res = prepare_backtest_predictions(
        res_final,
        score_smooth_window=args.score_smooth_window,
        score_smooth_method=args.score_smooth_method,
    )
    print("\nPortfolio construction...")
    print(
        f"  top_n={args.top_n} | rebal_freq={args.rebal_freq} | "
        f"score_smooth={args.score_smooth_method}:{args.score_smooth_window} | "
        f"no_trade_band={args.no_trade_band:.2f}"
    )

    print("\nSignal direction test...")
    bt_o = backtest_from_predictions(
        signal_res,
        top_n=args.top_n,
        rebal_freq=args.rebal_freq,
        rev=False,
        score_smooth_window=args.score_smooth_window,
        score_smooth_method=args.score_smooth_method,
        no_trade_band=args.no_trade_band,
    )
    bt_r = backtest_from_predictions(
        signal_res,
        top_n=args.top_n,
        rebal_freq=args.rebal_freq,
        rev=True,
        score_smooth_window=args.score_smooth_window,
        score_smooth_method=args.score_smooth_method,
        no_trade_band=args.no_trade_band,
    )
    use_rev = (1 + bt_r["net_return"]).cumprod().iloc[-1] > (1 + bt_o["net_return"]).cumprod().iloc[-1]
    bt_final = bt_r if use_rev else bt_o
    print(f'Using {"reversed" if use_rev else "original"} signal.')

    res_plot = signal_res.copy()
    res_plot["predicted"] = res_plot["trade_signal"]
    if use_rev:
        res_plot["predicted"] = -res_plot["predicted"]

    evaluate(bt_final, rebal_freq=args.rebal_freq)
    plot_fold_ic(metrics_df, res_final, run_paths["img_dir"])
    plot_backtest(bt_final, fold_metrics, res_plot, run_paths["img_dir"])
    signal_res_to_save = signal_res.copy()
    signal_res_to_save["trade_signal_active"] = -signal_res_to_save["trade_signal"] if use_rev else signal_res_to_save["trade_signal"]
    plot_daily_ic(res_final if not use_rev else res_final.assign(predicted=-res_final["predicted"]), run_paths["img_dir"])
    plot_stock_prediction_rolling(args.stock_plot, res_final if not use_rev else res_final.assign(predicted=-res_final["predicted"]), run_paths["img_dir"])

    metrics_df.to_csv(os.path.join(run_paths["run_dir"], "fold_metrics.csv"), index=False)
    signal_res_to_save.to_csv(os.path.join(run_paths["run_dir"], "predictions.csv"), index=False)
    bt_final.to_csv(os.path.join(run_paths["run_dir"], "backtest.csv"), index=False)
    print(f"[LOG] Saved tabular outputs into {run_paths['run_dir']}")


if __name__ == "__main__":
    main()
