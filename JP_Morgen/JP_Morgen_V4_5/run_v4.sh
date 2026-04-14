#!/bin/bash
# shellcheck disable=all

#SBATCH --job-name=JP_QuantV4_5
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a200:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load anaconda3/2024.06
source activate EricPy118

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

python main.py \
    --d_model 128 \
    --num_layers 3 \
    --nhead 4 \
    --dropout 0.15 \
    --listnet_weight 0.0 \
    --num_epochs 80 \
    --patience 10 \
    --lr 0.0003 \
    --batch_days 32 \
    --top_n 3 \
    --rebal_freq 5
