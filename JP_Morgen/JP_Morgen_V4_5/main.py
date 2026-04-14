# main.py
import os
import argparse
import pandas as pd
from datetime import datetime
import torch

from config import LOG_ROOT
from utils import setup_logging
from data_loader import process_and_normalize_data
from trainer import generate_folds, train_one_fold_v4
from backtest import backtest_from_predictions, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="JP Morgan Quant V4 Training on Cluster")
    
    # 架构参数
    parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dim')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of Two-Way blocks')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    
    # 损失函数参数
    parser.add_argument('--listnet_weight', type=float, default=0.0, help='Weight of ListNet Loss (0.0 means 100% IC)')
    parser.add_argument('--temperature', type=float, default=1.0, help='ListNet temp')
    
    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=60, help='Max epochs per fold')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_days', type=int, default=32, help='Days packaged into a batch')
    
    # 回测参数
    parser.add_argument('--top_n', type=int, default=3, help='Top N stocks')
    parser.add_argument('--rebal_freq', type=int, default=5, help='Rebalance frequency')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化日志与目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_name = f"V4_d{args.d_model}_l{args.num_layers}_lr{args.lr}_{timestamp}"
    run_dir = os.path.join(LOG_ROOT, run_name)
    img_dir = os.path.join(run_dir, "img")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    setup_logging(run_dir)
    print(f"{'='*60}\n  Run:  {run_name}\n  Dir:  {run_dir}\n{'='*60}")
    print(f"Hyperparameters: {vars(args)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 准备数据
    full_data = process_and_normalize_data()
    valid_dates_dt = [pd.Timestamp(d).to_pydatetime() for d in full_data['valid_dates']]
    date_range_years = (max(valid_dates_dt) - min(valid_dates_dt)).days / 365.25
    
    if date_range_years >= 7: mt, vm, tm = 3, 6, 6
    elif date_range_years >= 4: mt, vm, tm = 2, 4, 4
    else: mt, vm, tm = 1, 3, 3
        
    folds = generate_folds(valid_dates_dt, val_months=vm, test_months=tm, min_train_years=mt)
    print(f"\n{len(folds)} folds generated.")
    
    # 训练循环
    all_results = []
    fold_metrics = []
    for i, fold in enumerate(folds):
        print(f"\n{'='*60}\nFold {i+1}/{len(folds)}: test [{fold['test_start'].strftime('%Y-%m-%d')} ~ {fold['test_end'].strftime('%Y-%m-%d')}]\n{'='*60}")
        result, val_ic, test_ic = train_one_fold_v4(fold, full_data, device, args)
        if len(result) > 0:
            all_results.append(result)
            fold_metrics.append({'fold': i+1, 'test_start': fold['test_start'], 'val_ic': val_ic, 'test_ic': test_ic})
            
    if not all_results:
        print("No results generated.")
        return

    res_final = pd.concat(all_results, ignore_index=True).sort_values('date').reset_index(drop=True)
    
    # 信号方向与回测
    print('\nSignal direction test...')
    bt_o = backtest_from_predictions(res_final, top_n=args.top_n, rebal_freq=args.rebal_freq, rev=False)
    bt_r = backtest_from_predictions(res_final, top_n=args.top_n, rebal_freq=args.rebal_freq, rev=True)
    use_rev = (1+bt_r['net_return']).cumprod().iloc[-1] > (1+bt_o['net_return']).cumprod().iloc[-1]
    bt_final = bt_r if use_rev else bt_o
    print(f'Using {"reversed" if use_rev else "original"} signal.')
    
    evaluate(bt_final, rebal_freq=args.rebal_freq)

if __name__ == "__main__":
    main()