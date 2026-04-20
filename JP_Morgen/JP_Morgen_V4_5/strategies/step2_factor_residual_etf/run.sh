#!/bin/bash
# shellcheck disable=all

#SBATCH --job-name=JP_QuantV4_5_Step2ETFResidual
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:h200:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

preset="${1:-scale_4_277M}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
    echo "[run.sh] No active Slurm allocation detected."
    echo "[run.sh] Submitting this script with sbatch so it does not run on the login node CPU."
    sbatch "$0" "$preset" "$@"
    exit 0
fi

module load anaconda3/2024.06
source activate EricPy118

PROJECT_DIR="/home/wang.yige/7380/JP_Morgen_V4_5"
cd "$PROJECT_DIR"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

common_args=(
    --dropout 0.15
    --listnet_weight 0.0
    --num_epochs 100
    --patience 15
    --lr 0.0003
    --beta_window 120
    --beta_min_obs 60
    --top_n 5
    --rebal_freq 5
    --score_smooth_window 3
    --score_smooth_method ewm
    --no_trade_band 0.30
    --amp_mode on
    --amp_dtype float16
)

case "$preset" in
    fast)
        model_args=(
            --d_model 128
            --num_layers 3
            --nhead 4
            --batch_days 20
        )
        ;;
    scale_2_412M|medium)
        model_args=(
            --d_model 192
            --num_layers 4
            --nhead 6
            --batch_days 20
        )
        ;;
    scale_4_277M|large)
        model_args=(
            --d_model 256
            --num_layers 4
            --nhead 8
            --batch_days 16
        )
        ;;
    *)
        echo "Unknown preset: $preset" >&2
        echo "Usage: bash run.sh [fast|scale_2_412M|scale_4_277M] [extra args...]" >&2
        exit 1
        ;;
esac

echo "Project dir: $PROJECT_DIR"
echo "Preset: $preset"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: python strategies/step2_factor_residual_etf/main.py ${model_args[*]} ${common_args[*]} $*"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run.sh] nvidia-smi -L"
    nvidia-smi -L || true
fi

python - <<'PY'
import sys
import torch

print(f"[run.sh] torch.__version__ = {torch.__version__}")
print(f"[run.sh] cuda_available = {torch.cuda.is_available()}")
print(f"[run.sh] cuda_device_count = {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "[run.sh] ERROR: CUDA is unavailable in this job. "
        "Do not run training on the login node; use sbatch/srun with a GPU allocation."
    )
PY

python strategies/step2_factor_residual_etf/main.py \
    "${model_args[@]}" \
    "${common_args[@]}" \
    "$@"
