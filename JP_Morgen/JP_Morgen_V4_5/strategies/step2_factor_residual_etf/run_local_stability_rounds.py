import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STRATEGY_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import LOG_ROOT


MAIN_SCRIPT = STRATEGY_DIR / "main.py"
OFFLINE_SWEEP_SCRIPT = PROJECT_DIR / "evaluate_saved_predictions.py"
DIR_LINE_RE = re.compile(r"^\s*Dir:\s+(.*?)\s*$")
DEFAULT_SEEDS = [101, 202, 303, 404, 505]
ROUND_CONFIGS = {
    "round1_base": {
        "description": "Round 1 baseline stability sweep with the simple trading layer fixed.",
        "main_args": [
            "--d_model",
            "256",
            "--num_layers",
            "4",
            "--nhead",
            "8",
            "--dropout",
            "0.15",
            "--listnet_weight",
            "0.0",
            "--num_epochs",
            "100",
            "--patience",
            "15",
            "--lr",
            "0.0003",
            "--weight_decay",
            "0.0001",
            "--grad_clip_norm",
            "1.0",
            "--early_stop_min_delta",
            "0.0",
            "--batch_days",
            "16",
            "--beta_window",
            "120",
            "--beta_min_obs",
            "60",
            "--top_n",
            "3",
            "--rebal_freq",
            "5",
            "--score_smooth_window",
            "1",
            "--score_smooth_method",
            "off",
            "--no_trade_band",
            "0.0",
            "--amp_mode",
            "on",
            "--amp_dtype",
            "float16",
        ],
    },
    "round2_regularized": {
        "description": "Round 2 regularized stability sweep with stronger dropout, weight decay, and early-stop delta.",
        "main_args": [
            "--d_model",
            "256",
            "--num_layers",
            "4",
            "--nhead",
            "8",
            "--dropout",
            "0.20",
            "--listnet_weight",
            "0.0",
            "--num_epochs",
            "100",
            "--patience",
            "15",
            "--lr",
            "0.0003",
            "--weight_decay",
            "0.0005",
            "--grad_clip_norm",
            "1.0",
            "--early_stop_min_delta",
            "0.001",
            "--batch_days",
            "16",
            "--beta_window",
            "120",
            "--beta_min_obs",
            "60",
            "--top_n",
            "3",
            "--rebal_freq",
            "5",
            "--score_smooth_window",
            "1",
            "--score_smooth_method",
            "off",
            "--no_trade_band",
            "0.0",
            "--amp_mode",
            "on",
            "--amp_dtype",
            "float16",
        ],
    },
}


def parse_seed_list(text):
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def sanitize_extra_args(extra_args):
    extra = list(extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_selected_direction(predictions_path):
    if not predictions_path.exists():
        return None

    frame = pd.read_csv(predictions_path, usecols=lambda column: column in {"trade_signal", "trade_signal_active"})
    if "trade_signal" not in frame.columns or "trade_signal_active" not in frame.columns:
        return None
    if frame.empty:
        return None

    same_gap = (frame["trade_signal_active"] - frame["trade_signal"]).abs().mean()
    flip_gap = (frame["trade_signal_active"] + frame["trade_signal"]).abs().mean()
    return "reversed" if flip_gap < same_gap else "original"


def summarize_backtest(backtest_path):
    if not backtest_path.exists():
        return {}

    frame = pd.read_csv(backtest_path)
    if frame.empty:
        return {
            "gross_total": np.nan,
            "net_total": np.nan,
            "avg_turnover": np.nan,
            "net_max_drawdown": np.nan,
            "n_backtest_rows": 0,
        }

    gross_curve = (1.0 + frame["gross_return"]).cumprod()
    net_curve = (1.0 + frame["net_return"]).cumprod()
    max_drawdown = ((net_curve - net_curve.cummax()) / net_curve.cummax()).min()
    return {
        "gross_total": float(gross_curve.iloc[-1] - 1.0),
        "net_total": float(net_curve.iloc[-1] - 1.0),
        "avg_turnover": float(frame["turnover"].mean()),
        "net_max_drawdown": float(max_drawdown),
        "n_backtest_rows": int(len(frame)),
    }


def summarize_fold_metrics(fold_metrics_path):
    if not fold_metrics_path.exists():
        return {}

    frame = pd.read_csv(fold_metrics_path)
    if frame.empty:
        return {
            "avg_val_ic": np.nan,
            "avg_test_ic": np.nan,
            "ic_gap": np.nan,
            "mean_best_epoch": np.nan,
            "median_best_epoch": np.nan,
            "n_folds": 0,
        }

    avg_val_ic = float(frame["val_ic"].mean())
    avg_test_ic = float(frame["test_ic"].mean())
    return {
        "avg_val_ic": avg_val_ic,
        "avg_test_ic": avg_test_ic,
        "ic_gap": float(avg_val_ic - avg_test_ic),
        "mean_best_epoch": float(frame["best_epoch"].dropna().mean()) if frame["best_epoch"].notna().any() else np.nan,
        "median_best_epoch": float(frame["best_epoch"].dropna().median()) if frame["best_epoch"].notna().any() else np.nan,
        "n_folds": int(len(frame)),
    }


def summarize_run_dir(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest.get("parameters", {})
    fold_metrics = summarize_fold_metrics(run_dir / "fold_metrics.csv")
    backtest_metrics = summarize_backtest(run_dir / "backtest.csv")

    row = {
        "run_name": manifest.get("artifacts", {}).get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "seed": parameters.get("seed"),
        "deterministic": parameters.get("deterministic"),
        "d_model": parameters.get("d_model"),
        "num_layers": parameters.get("num_layers"),
        "nhead": parameters.get("nhead"),
        "dropout": parameters.get("dropout"),
        "lr": parameters.get("lr"),
        "weight_decay": parameters.get("weight_decay"),
        "grad_clip_norm": parameters.get("grad_clip_norm"),
        "early_stop_min_delta": parameters.get("early_stop_min_delta"),
        "batch_days": parameters.get("batch_days"),
        "top_n": parameters.get("top_n"),
        "rebal_freq": parameters.get("rebal_freq"),
        "score_smooth_method": parameters.get("score_smooth_method"),
        "score_smooth_window": parameters.get("score_smooth_window"),
        "no_trade_band": parameters.get("no_trade_band"),
        "selected_direction": infer_selected_direction(run_dir / "predictions.csv"),
    }
    row.update(fold_metrics)
    row.update(backtest_metrics)
    return row


def aggregate_run_summary(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    config_cols = [
        "d_model",
        "num_layers",
        "nhead",
        "dropout",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "early_stop_min_delta",
        "batch_days",
        "top_n",
        "rebal_freq",
        "score_smooth_method",
        "score_smooth_window",
        "no_trade_band",
        "deterministic",
    ]
    rows = []
    grouped = summary_df.groupby(config_cols, dropna=False, sort=False)
    for config_values, group in grouped:
        row = {column: value for column, value in zip(config_cols, config_values)}
        seeds = group["seed"].dropna().astype(int).tolist()
        row.update(
            {
                "n_runs": int(len(group)),
                "seeds": ",".join(str(seed) for seed in sorted(seeds)),
                "mean_val_ic": float(group["avg_val_ic"].mean()),
                "mean_test_ic": float(group["avg_test_ic"].mean()),
                "median_test_ic": float(group["avg_test_ic"].median()),
                "std_test_ic": float(group["avg_test_ic"].std(ddof=0)) if len(group) > 1 else 0.0,
                "mean_ic_gap": float(group["ic_gap"].mean()),
                "mean_net_total": float(group["net_total"].mean()),
                "median_net_total": float(group["net_total"].median()),
                "positive_net_runs": int((group["net_total"] > 0).sum()),
                "positive_net_rate": float((group["net_total"] > 0).mean()),
                "original_direction_runs": int((group["selected_direction"] == "original").sum()),
                "reversed_direction_runs": int((group["selected_direction"] == "reversed").sum()),
                "mean_best_epoch": float(group["mean_best_epoch"].mean()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["mean_test_ic", "mean_net_total"],
        ascending=[False, False],
    ).reset_index(drop=True)


def write_round_summary(round_dir, run_dirs):
    summary_dir = round_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run_dir(run_dir) for run_dir in run_dirs]
    summary_df = pd.DataFrame(rows).sort_values(["seed", "run_name"]).reset_index(drop=True)
    aggregate_df = aggregate_run_summary(summary_df)

    run_summary_path = summary_dir / "run_summary.csv"
    aggregate_path = summary_dir / "aggregate_summary.csv"
    manifest_path = summary_dir / "summary_manifest.json"

    summary_df.to_csv(run_summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    write_json(
        manifest_path,
        {
            "run_dirs": [str(Path(run_dir).resolve()) for run_dir in run_dirs],
            "files": {
                "run_summary_csv": str(run_summary_path),
                "aggregate_summary_csv": str(aggregate_path),
            },
        },
    )
    return {
        "run_summary_csv": str(run_summary_path),
        "aggregate_summary_csv": str(aggregate_path),
        "summary_manifest_json": str(manifest_path),
    }


def find_recent_run_dir(seed, started_at):
    log_root = Path(LOG_ROOT)
    if not log_root.exists():
        return None

    seed_token = f"seed{seed}"
    threshold = started_at - timedelta(minutes=10)
    candidates = []
    for path in log_root.iterdir():
        if not path.is_dir():
            continue
        if seed_token not in path.name:
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at >= threshold:
            candidates.append(path)

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item.stat().st_mtime).resolve())


def run_training_command(command, console_log_path, seed):
    started_at = datetime.now(timezone.utc)
    run_dir = None
    console_log_path.parent.mkdir(parents=True, exist_ok=True)

    with console_log_path.open("w", encoding="utf-8") as console_log:
        process = subprocess.Popen(
            command,
            cwd=str(STRATEGY_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            console_log.write(line)
            if run_dir is None:
                match = DIR_LINE_RE.search(line)
                if match:
                    run_dir = match.group(1).strip().rstrip("/\\")
        process.wait()

    if run_dir is None:
        run_dir = find_recent_run_dir(seed=seed, started_at=started_at)

    return {
        "returncode": int(process.returncode),
        "run_dir": run_dir,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "console_log": str(console_log_path.resolve()),
        "command": command,
    }


def run_offline_sweep(python_exe, round_dir, run_dirs):
    output_dir = round_dir / "offline_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [python_exe, "-u", str(OFFLINE_SWEEP_SCRIPT), *run_dirs, "--output_dir", str(output_dir)]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    sweep_log = round_dir / "offline_sweep.log"
    sweep_log.write_text(
        (completed.stdout or "") + ("\n" if completed.stdout else "") + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Offline sweep failed for {round_dir}: {completed.stderr}")
    return {
        "output_dir": str(output_dir.resolve()),
        "log_file": str(sweep_log.resolve()),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run local Step 2 stability rounds in Quant311.")
    parser.add_argument(
        "--round",
        choices=["all", "round1_base", "round2_regularized"],
        default="all",
        help="Which round to run. 'all' runs round 1 first, then round 2.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used for training and offline sweep.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional directory used to store round manifests and summaries.",
    )
    parser.add_argument(
        "--skip_offline_sweep",
        action="store_true",
        help="Skip evaluate_saved_predictions.py after each completed round.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to the next seed after a failed run instead of stopping immediately.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to strategies/step2_factor_residual_etf/main.py. Use '--' before them.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    python_exe = str(Path(args.python_exe).expanduser().resolve())
    seeds = parse_seed_list(args.seeds)
    extra_args = sanitize_extra_args(args.extra_args)
    rounds = list(ROUND_CONFIGS) if args.round == "all" else [args.round]

    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else Path(LOG_ROOT).resolve() / "rounds" / f"local_stability_rounds__{session_timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    session_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_exe": python_exe,
        "round_request": args.round,
        "seeds": seeds,
        "output_root": str(output_root),
        "skip_offline_sweep": bool(args.skip_offline_sweep),
        "continue_on_error": bool(args.continue_on_error),
        "extra_args": extra_args,
        "rounds": [],
    }
    session_manifest_path = output_root / "session_manifest.json"
    write_json(session_manifest_path, session_manifest)

    for round_name in rounds:
        round_config = ROUND_CONFIGS[round_name]
        round_dir = output_root / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        round_manifest = {
            "round_name": round_name,
            "description": round_config["description"],
            "round_dir": str(round_dir),
            "python_exe": python_exe,
            "seeds": seeds,
            "extra_args": extra_args,
            "runs": [],
            "summary": None,
            "offline_sweep": None,
        }
        round_manifest_path = round_dir / "round_manifest.json"
        write_json(round_manifest_path, round_manifest)

        print("")
        print("=" * 80)
        print(f"Starting {round_name}")
        print(round_config["description"])
        print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
        print("=" * 80)

        successful_run_dirs = []
        for seed in seeds:
            command = [
                python_exe,
                "-u",
                str(MAIN_SCRIPT),
                *round_config["main_args"],
                "--seed",
                str(seed),
                "--deterministic",
                *extra_args,
            ]
            console_log_path = round_dir / "seed_logs" / f"seed{seed}.console.log"
            result = run_training_command(command, console_log_path=console_log_path, seed=seed)
            result["seed"] = seed
            round_manifest["runs"].append(result)
            write_json(round_manifest_path, round_manifest)

            if result["returncode"] != 0:
                print(f"[round] seed {seed} failed with exit code {result['returncode']}")
                if not args.continue_on_error:
                    raise RuntimeError(f"{round_name} failed on seed {seed}. See {console_log_path}")
            elif result["run_dir"]:
                successful_run_dirs.append(result["run_dir"])

        if successful_run_dirs:
            round_manifest["summary"] = write_round_summary(round_dir, successful_run_dirs)
            write_json(round_manifest_path, round_manifest)
            if not args.skip_offline_sweep:
                round_manifest["offline_sweep"] = run_offline_sweep(
                    python_exe=python_exe,
                    round_dir=round_dir,
                    run_dirs=successful_run_dirs,
                )
                write_json(round_manifest_path, round_manifest)

        session_manifest["rounds"].append(round_manifest)
        write_json(session_manifest_path, session_manifest)

    print("")
    print("=" * 80)
    print(f"Completed requested local stability work. Session dir: {output_root}")
    print(f"Session manifest: {session_manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
