#!/bin/bash
#SBATCH --job-name=JP_QuantV4
#SBATCH --output=v4_%j.out
#SBATCH --error=v4_%j.err
#SBATCH --partition=gpu           # 替换为你们集群的 GPU 队列名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # CPU 核心数
#SBATCH --gres=gpu:1              # 申请 1 张 GPU
#SBATCH --mem=32G                 # 内存大小
#SBATCH --time=12:00:00           # 预计运行时间

# 加载环境 (替换为你集群的 conda/环境配置)
# module load anaconda3
# module load cuda/11.8
# source activate quant_env

# 运行主程序，你可以轻松在这里修改参数而不用动代码
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