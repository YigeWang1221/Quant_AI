$python = "C:\Users\59386\.conda\envs\Quant311\python.exe"
$script = "D:\quant\quant1\JP_Morgen\JP_Morgen_V4_5\strategies\step2_factor_residual_etf\main.py"

& $python $script `
  --d_model 256 `
  --num_layers 4 `
  --nhead 8 `
  --dropout 0.15 `
  --listnet_weight 0.0 `
  --num_epochs 100 `
  --patience 15 `
  --lr 0.0003 `
  --batch_days 16 `
  --beta_window 120 `
  --beta_min_obs 60 `
  --top_n 3 `
  --rebal_freq 5 `
  --amp_mode on `
  --amp_dtype float16
