import os
import argparse
import pandas as pd

from config import (
    DEFAULT_AMP_DTYPE,
    DEFAULT_AMP_MODE,
    DEFAULT_BATCH_DAYS,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_LISTNET_WEIGHT,
    DEFAULT_LR,
    DEFAULT_NHEAD,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    DEFAULT_REBAL_FREQ,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_N,
)
from dataset import process_and_normalize_data
from trainer import describe_folds, generate_folds, train_one_fold_v4
from backtest import backtest_from_predictions, evaluate
from utils import create_run_dirs, get_device, setup_logging
from visualization import plot_backtest, plot_daily_ic, plot_fold_ic, plot_stock_prediction_rolling

def parse_args():
    parser = argparse.ArgumentParser(description="JP Morgan Quant V4.5 Training")
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL, help="Transformer hidden dim")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of Two-Way blocks")
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--listnet_weight", type=float, default=DEFAULT_LISTNET_WEIGHT, help="Weight of ListNet Loss")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="ListNet temperature")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--batch_days", type=int, default=DEFAULT_BATCH_DAYS, help="Days packaged into a batch")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="Top N stocks")
    parser.add_argument("--rebal_freq", type=int, default=DEFAULT_REBAL_FREQ, help="Rebalance frequency")
    parser.add_argument("--stock_plot", type=str, default="COP", help="Ticker to visualize")
    parser.add_argument(
        "--amp_mode",
        type=str,
        default=DEFAULT_AMP_MODE,
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


def main():
    args = parse_args()

    run_label = f"d{args.d_model}_l{args.num_layers}_lr{args.lr}"
    run_paths = create_run_dirs(run_label)
    setup_logging(run_paths)
    print(f"Hyperparameters: {vars(args)}")

    device = get_device()
    print(f"Device: {device}")
    args.amp_enabled = device.type == "cuda" and args.amp_mode != "off"
    if args.amp_mode == "on" and device.type != "cuda":
        print("AMP requested, but CUDA is unavailable. Falling back to full precision.")
    print(f"Forward AMP: {'on' if args.amp_enabled else 'off'} ({args.amp_dtype}) | Loss precision: fp32")

    full_data = process_and_normalize_data()
    valid_dates_dt = [pd.Timestamp(d).to_pydatetime() for d in full_data['valid_dates']]
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
        print(f"\n{'='*60}\nFold {i+1}/{len(folds)}: test [{fold['test_start'].strftime('%Y-%m-%d')} ~ {fold['test_end'].strftime('%Y-%m-%d')}]\n{'='*60}")
        result, val_ic, test_ic = train_one_fold_v4(fold, full_data, device, args)
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
    print(f"\n{'='*60}\nV4 Rolling Window Summary\n{'='*60}")
    for _, row in metrics_df.iterrows():
        print(f"  Fold {row['fold']}: Val IC={row['val_ic']:.4f} | Test IC={row['test_ic']:.4f} | [{row['test_start'].strftime('%Y-%m')} ~ {row['test_end'].strftime('%Y-%m')}]")
    print(f"\n  Avg Val IC:  {metrics_df['val_ic'].mean():.4f}")
    print(f"  Avg Test IC: {metrics_df['test_ic'].mean():.4f}")
    print(f"  IC gap: {metrics_df['val_ic'].mean() - metrics_df['test_ic'].mean():.4f}")
    print(f"  Total predictions: {len(res_final):,}")

    print("\nSignal direction test...")
    bt_o = backtest_from_predictions(res_final, top_n=args.top_n, rebal_freq=args.rebal_freq, rev=False)
    bt_r = backtest_from_predictions(res_final, top_n=args.top_n, rebal_freq=args.rebal_freq, rev=True)
    use_rev = (1 + bt_r["net_return"]).cumprod().iloc[-1] > (1 + bt_o["net_return"]).cumprod().iloc[-1]
    bt_final = bt_r if use_rev else bt_o
    print(f'Using {"reversed" if use_rev else "original"} signal.')

    res_plot = res_final.copy()
    if use_rev:
        res_plot["predicted"] = -res_plot["predicted"]

    evaluate(bt_final, rebal_freq=args.rebal_freq)
    plot_fold_ic(metrics_df, res_final, run_paths["img_dir"])
    plot_backtest(bt_final, fold_metrics, res_plot, run_paths["img_dir"])
    plot_daily_ic(res_plot, run_paths["img_dir"])
    plot_stock_prediction_rolling(args.stock_plot, res_plot, run_paths["img_dir"])

    metrics_df.to_csv(os.path.join(run_paths["run_dir"], "fold_metrics.csv"), index=False)
    res_final.to_csv(os.path.join(run_paths["run_dir"], "predictions.csv"), index=False)
    bt_final.to_csv(os.path.join(run_paths["run_dir"], "backtest.csv"), index=False)
    print(f"[LOG] Saved tabular outputs into {run_paths['run_dir']}")


if __name__ == "__main__":
    main()
