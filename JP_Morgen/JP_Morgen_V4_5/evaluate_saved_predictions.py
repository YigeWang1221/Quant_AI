import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import backtest_from_predictions, prepare_backtest_predictions


BASE_COLUMNS = ["ticker", "date", "predicted", "actual", "raw_actual"]
DEFAULT_TOP_N_VALUES = [3, 5]
DEFAULT_REBAL_FREQ_VALUES = [5, 10]
DEFAULT_SCORE_SMOOTH_METHODS = ["off", "ewm"]
DEFAULT_SCORE_SMOOTH_WINDOWS = [1, 3]
DEFAULT_NO_TRADE_BAND_VALUES = [0.0, 0.30]


def parse_int_list(text):
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_float_list(text):
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_str_list(text):
    return [token.strip() for token in str(text).split(",") if token.strip()]


def resolve_predictions_paths(inputs):
    resolved = []
    seen = set()

    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        candidates = []
        if path.is_file():
            if path.name != "predictions.csv":
                raise ValueError(f"Expected a predictions.csv file, got: {path}")
            candidates = [path]
        else:
            direct = path / "predictions.csv"
            if direct.exists():
                candidates = [direct]
            else:
                candidates = sorted(path.rglob("predictions.csv"))

        for candidate in candidates:
            key = str(candidate.resolve())
            if key not in seen:
                resolved.append(candidate.resolve())
                seen.add(key)

    if not resolved:
        raise ValueError("No predictions.csv files were found under the provided inputs.")
    return resolved


def iter_sweep_configs(top_ns, rebal_freqs, smooth_methods, smooth_windows, no_trade_bands):
    configs = []
    for top_n, rebal_freq, smooth_method, smooth_window, no_trade_band in product(
        top_ns,
        rebal_freqs,
        smooth_methods,
        smooth_windows,
        no_trade_bands,
    ):
        method = smooth_method.lower()
        if method == "off" and smooth_window != 1:
            continue
        configs.append(
            {
                "top_n": top_n,
                "rebal_freq": rebal_freq,
                "score_smooth_method": method,
                "score_smooth_window": smooth_window,
                "no_trade_band": no_trade_band,
            }
        )
    return configs


def summarize_backtest(bt, rebal_freq):
    if bt.empty:
        return {
            "periods": 0,
            "gross_total": np.nan,
            "net_total": np.nan,
            "gross_annual": np.nan,
            "net_annual": np.nan,
            "net_vol": np.nan,
            "net_sharpe": np.nan,
            "net_max_drawdown": np.nan,
            "net_win_rate": np.nan,
            "avg_turnover": np.nan,
        }

    periods_per_year = 252 / rebal_freq
    gross_curve = (1.0 + bt["gross_return"]).cumprod()
    net_curve = (1.0 + bt["net_return"]).cumprod()

    gross_total = gross_curve.iloc[-1] - 1.0
    net_total = net_curve.iloc[-1] - 1.0
    periods = len(bt)
    gross_annual = (1.0 + gross_total) ** (periods_per_year / periods) - 1.0
    net_annual = (1.0 + net_total) ** (periods_per_year / periods) - 1.0
    net_vol = bt["net_return"].std() * np.sqrt(periods_per_year)
    net_sharpe = (net_annual - 0.04) / net_vol if net_vol > 0 else 0.0
    net_max_drawdown = ((net_curve - net_curve.cummax()) / net_curve.cummax()).min()
    net_win_rate = (bt["net_return"] > 0).mean()
    avg_turnover = bt["turnover"].mean()

    return {
        "periods": periods,
        "gross_total": gross_total,
        "net_total": net_total,
        "gross_annual": gross_annual,
        "net_annual": net_annual,
        "net_vol": net_vol,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": net_max_drawdown,
        "net_win_rate": net_win_rate,
        "avg_turnover": avg_turnover,
    }


def summarize_by_year(bt):
    if bt.empty:
        return []

    frame = bt.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows = []
    for year, grp in frame.groupby("year"):
        net_curve = (1.0 + grp["net_return"]).cumprod()
        rows.append(
            {
                "year": int(year),
                "year_net_total": net_curve.iloc[-1] - 1.0,
                "year_avg_turnover": grp["turnover"].mean(),
                "year_periods": len(grp),
            }
        )
    return rows


def pick_output_dir(prediction_paths, output_dir_arg):
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()

    if len(prediction_paths) == 1:
        return prediction_paths[0].parent / "offline_sweep"
    return Path(__file__).resolve().parent / "offline_sweep_results"


def prepare_base_predictions(path):
    frame = pd.read_csv(path)
    available = [column for column in BASE_COLUMNS if column in frame.columns]
    missing = {"ticker", "date", "predicted", "actual"} - set(available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[available].copy()


def run_sweep_for_predictions(path, configs, direction_mode):
    base = prepare_base_predictions(path)
    prepared_cache = {}
    sweep_rows = []
    yearly_rows = []
    run_name = path.parent.name

    for config in configs:
        cache_key = (config["score_smooth_method"], config["score_smooth_window"])
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_backtest_predictions(
                base,
                score_smooth_window=config["score_smooth_window"],
                score_smooth_method=config["score_smooth_method"],
            )
        prepared = prepared_cache[cache_key]

        candidate_metrics = []
        direction_candidates = [False, True] if direction_mode == "auto" else [direction_mode == "reversed"]
        for rev in direction_candidates:
            bt = backtest_from_predictions(
                prepared,
                top_n=config["top_n"],
                rebal_freq=config["rebal_freq"],
                rev=rev,
                score_smooth_window=config["score_smooth_window"],
                score_smooth_method=config["score_smooth_method"],
                no_trade_band=config["no_trade_band"],
            )
            metrics = summarize_backtest(bt, rebal_freq=config["rebal_freq"])
            metrics["direction"] = "reversed" if rev else "original"
            metrics["backtest"] = bt
            candidate_metrics.append(metrics)

        best_metrics = max(candidate_metrics, key=lambda item: item["net_total"])
        original_net = next((item["net_total"] for item in candidate_metrics if item["direction"] == "original"), np.nan)
        reversed_net = next((item["net_total"] for item in candidate_metrics if item["direction"] == "reversed"), np.nan)

        summary_row = {
            "run_name": run_name,
            "predictions_path": str(path),
            **config,
            "selected_direction": best_metrics["direction"],
            "original_net_total": original_net,
            "reversed_net_total": reversed_net,
            "direction_net_gap": np.nan if np.isnan(original_net) or np.isnan(reversed_net) else reversed_net - original_net,
            "periods": best_metrics["periods"],
            "gross_total": best_metrics["gross_total"],
            "net_total": best_metrics["net_total"],
            "gross_annual": best_metrics["gross_annual"],
            "net_annual": best_metrics["net_annual"],
            "net_vol": best_metrics["net_vol"],
            "net_sharpe": best_metrics["net_sharpe"],
            "net_max_drawdown": best_metrics["net_max_drawdown"],
            "net_win_rate": best_metrics["net_win_rate"],
            "avg_turnover": best_metrics["avg_turnover"],
        }
        sweep_rows.append(summary_row)

        for row in summarize_by_year(best_metrics["backtest"]):
            yearly_rows.append(
                {
                    "run_name": run_name,
                    "predictions_path": str(path),
                    **config,
                    "selected_direction": best_metrics["direction"],
                    **row,
                }
            )

    summary = pd.DataFrame(sweep_rows).sort_values(["run_name", "net_total"], ascending=[True, False]).reset_index(drop=True)
    yearly = pd.DataFrame(yearly_rows)
    summary["rank_within_run"] = summary.groupby("run_name")["net_total"].rank(method="dense", ascending=False).astype(int)
    return summary, yearly


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Offline portfolio sweep over saved predictions.csv files.")
    parser.add_argument("inputs", nargs="+", help="Run directories, directory roots, or predictions.csv files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory used to save sweep outputs.")
    parser.add_argument(
        "--top_n_values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_TOP_N_VALUES),
        help="Comma-separated top_n values.",
    )
    parser.add_argument(
        "--rebal_freq_values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_REBAL_FREQ_VALUES),
        help="Comma-separated rebalance frequencies.",
    )
    parser.add_argument(
        "--score_smooth_methods",
        type=str,
        default=",".join(DEFAULT_SCORE_SMOOTH_METHODS),
        help="Comma-separated score smoothing methods.",
    )
    parser.add_argument(
        "--score_smooth_windows",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SCORE_SMOOTH_WINDOWS),
        help="Comma-separated score smoothing windows.",
    )
    parser.add_argument(
        "--no_trade_band_values",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_NO_TRADE_BAND_VALUES),
        help="Comma-separated no-trade-band values.",
    )
    parser.add_argument(
        "--direction_mode",
        choices=["auto", "original", "reversed"],
        default="auto",
        help="Test both directions and keep the better one, or force a fixed direction.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top rows to print per run.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    prediction_paths = resolve_predictions_paths(args.inputs)
    configs = iter_sweep_configs(
        top_ns=parse_int_list(args.top_n_values),
        rebal_freqs=parse_int_list(args.rebal_freq_values),
        smooth_methods=parse_str_list(args.score_smooth_methods),
        smooth_windows=parse_int_list(args.score_smooth_windows),
        no_trade_bands=parse_float_list(args.no_trade_band_values),
    )
    if not configs:
        raise ValueError("The sweep grid is empty after filtering invalid combinations.")

    output_dir = pick_output_dir(prediction_paths, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    yearly_frames = []
    for path in prediction_paths:
        summary, yearly = run_sweep_for_predictions(path, configs=configs, direction_mode=args.direction_mode)
        summary_frames.append(summary)
        if not yearly.empty:
            yearly_frames.append(yearly)

    summary_df = pd.concat(summary_frames, ignore_index=True).sort_values(["run_name", "net_total"], ascending=[True, False])
    best_df = summary_df.sort_values(["run_name", "rank_within_run"]).groupby("run_name", as_index=False).head(1).reset_index(drop=True)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

    summary_path = output_dir / "sweep_summary.csv"
    best_path = output_dir / "best_config_by_run.csv"
    yearly_path = output_dir / "best_config_yearly.csv"
    manifest_path = output_dir / "sweep_manifest.json"

    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)
    if not yearly_df.empty:
        best_keys = best_df[
            [
                "run_name",
                "top_n",
                "rebal_freq",
                "score_smooth_method",
                "score_smooth_window",
                "no_trade_band",
                "selected_direction",
            ]
        ]
        best_yearly = yearly_df.merge(best_keys, on=list(best_keys.columns), how="inner")
        best_yearly.to_csv(yearly_path, index=False)

    manifest = {
        "inputs": [str(path) for path in prediction_paths],
        "output_dir": str(output_dir),
        "direction_mode": args.direction_mode,
        "grid": configs,
        "files": {
            "sweep_summary": str(summary_path),
            "best_config_by_run": str(best_path),
            "best_config_yearly": str(yearly_path) if not yearly_df.empty else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Resolved {len(prediction_paths)} predictions files.")
    print(f"Evaluated {len(configs)} sweep configs per file.")
    print(f"Saved summary: {summary_path}")
    print(f"Saved best-by-run: {best_path}")
    if not yearly_df.empty:
        print(f"Saved yearly best summary: {yearly_path}")
    print("")

    for run_name, group in summary_df.groupby("run_name"):
        print(f"Run: {run_name}")
        cols = [
            "rank_within_run",
            "top_n",
            "rebal_freq",
            "score_smooth_method",
            "score_smooth_window",
            "no_trade_band",
            "selected_direction",
            "net_total",
            "avg_turnover",
            "net_max_drawdown",
        ]
        print(group.head(args.top_k)[cols].to_string(index=False))
        print("")


if __name__ == "__main__":
    main()
