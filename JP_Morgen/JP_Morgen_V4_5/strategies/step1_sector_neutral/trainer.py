import copy
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dateutil.relativedelta import relativedelta
from scipy import stats

from loss import V4CombinedLoss
from model import QuantV4
from utils import set_random_seed


def generate_folds(dates, val_months, test_months, min_train_years):
    unique_dates = sorted(set(pd.Timestamp(d).to_pydatetime() for d in dates))
    first_date, last_date = unique_dates[0], unique_dates[-1]
    folds = []
    test_start = first_date + relativedelta(years=min_train_years, months=val_months)

    while True:
        test_end = test_start + relativedelta(months=test_months)
        if test_end > last_date:
            if test_start < last_date:
                test_end = last_date
            else:
                break

        val_start = test_start - relativedelta(months=val_months)
        train_end = val_start
        train_dates = [d for d in unique_dates if first_date <= d < train_end]
        val_dates = [d for d in unique_dates if val_start <= d < test_start]
        test_dates = [d for d in unique_dates if test_start <= d < test_end]

        if len(train_dates) > 100 and len(val_dates) > 20 and len(test_dates) > 20:
            folds.append(
                {
                    "train_start": first_date,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": test_start,
                    "test_start": test_start,
                    "test_end": test_end,
                    "n_train": len(train_dates),
                    "n_val": len(val_dates),
                    "n_test": len(test_dates),
                }
            )

        test_start = test_end
        if test_end >= last_date:
            break

    return folds


def get_dates_in_range(all_dates, start, end):
    start_dt = pd.Timestamp(start).to_pydatetime()
    end_dt = pd.Timestamp(end).to_pydatetime()
    return [d for d in all_dates if start_dt <= pd.Timestamp(d).to_pydatetime() < end_dt]


def describe_folds(folds):
    print(f"{len(folds)} folds")
    for i, fold in enumerate(folds, start=1):
        print(
            f"  Fold {i}: Train [{fold['train_start'].strftime('%Y-%m')}~{fold['train_end'].strftime('%Y-%m')}] "
            f"Val [{fold['val_start'].strftime('%Y-%m')}~{fold['val_end'].strftime('%Y-%m')}] "
            f"Test [{fold['test_start'].strftime('%Y-%m')}~{fold['test_end'].strftime('%Y-%m')}] "
            f"({fold['n_train']}/{fold['n_val']}/{fold['n_test']})"
        )


def _pad_day_sample(sample, max_stocks):
    x = torch.FloatTensor(sample["X"])
    y = torch.FloatTensor(sample["y"])
    n_stocks = x.shape[0]
    if n_stocks < max_stocks:
        x = torch.cat([x, torch.zeros(max_stocks - n_stocks, x.shape[1], x.shape[2])])
        y = torch.cat([y, torch.zeros(max_stocks - n_stocks)])
    mask = torch.zeros(max_stocks)
    mask[:n_stocks] = 1.0
    return x, y, mask


def _amp_context(amp_enabled, amp_dtype):
    if not amp_enabled:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _move_to_device(tensor, device):
    if device.type == "cuda":
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def _stack_batch(batch_samples):
    xs, ys, masks = zip(*batch_samples)
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def _batch_ic_only_loss(pred_batch, y_batch, mask_batch, ic_weight):
    valid = mask_batch > 0
    valid_counts = valid.sum(dim=1)
    valid_rows = valid_counts >= 5
    if not torch.any(valid_rows):
        return pred_batch.new_zeros(()), 0.0

    valid_f = valid.float()
    safe_counts = valid_counts.clamp_min(1).to(pred_batch.dtype)
    pred_sum = (pred_batch * valid_f).sum(dim=1)
    target_sum = (y_batch * valid_f).sum(dim=1)
    pred_centered = (pred_batch - pred_sum.unsqueeze(1) / safe_counts.unsqueeze(1)) * valid_f
    target_centered = (y_batch - target_sum.unsqueeze(1) / safe_counts.unsqueeze(1)) * valid_f

    sample_denom = (valid_counts - 1).clamp_min(1).to(pred_batch.dtype)
    pred_std = torch.sqrt(pred_centered.square().sum(dim=1) / sample_denom + 1e-8)
    target_std = torch.sqrt(target_centered.square().sum(dim=1) / sample_denom + 1e-8)
    cov = (pred_centered * target_centered).sum(dim=1) / safe_counts
    ic = cov / (pred_std * target_std + 1e-8)

    loss = ic_weight * (1 - ic[valid_rows]).mean()
    mean_ic = ic[valid_rows].mean().item()
    return loss, mean_ic


def _batch_loss_and_ic(pred_batch, y_batch, mask_batch, criterion):
    if criterion.lw == 0:
        return _batch_ic_only_loss(pred_batch, y_batch, mask_batch, criterion.iw)

    total_loss = pred_batch.new_zeros(())
    total_ic = 0.0
    batch_size = pred_batch.shape[0]
    for i in range(batch_size):
        loss_i, _, ic_i = criterion(pred_batch[i], y_batch[i], mask_batch[i])
        total_loss = total_loss + loss_i
        total_ic += ic_i
    return total_loss / batch_size, total_ic / batch_size


def _predict_by_date(model, daily, dates, max_stocks, device, batch_days, amp_enabled, amp_dtype):
    predictions = {}
    for batch_start in range(0, len(dates), batch_days):
        batch_dates = dates[batch_start:batch_start + batch_days]
        x_cpu, _, mask_cpu = _stack_batch([_pad_day_sample(daily[dt], max_stocks) for dt in batch_dates])
        x_batch = _move_to_device(x_cpu, device)
        mask_batch = _move_to_device(mask_cpu, device)

        with _amp_context(amp_enabled, amp_dtype):
            pred_batch = model(x_batch, stock_mask=mask_batch)
        pred_batch = pred_batch.float().cpu()
        valid_counts = mask_cpu.sum(dim=1).int().tolist()

        for i, dt in enumerate(batch_dates):
            predictions[dt] = pred_batch[i, :valid_counts[i]].numpy()

        del x_batch, mask_batch, pred_batch, x_cpu, mask_cpu

    return predictions


def _resolve_fold_seed(args, fold_index):
    base_seed = getattr(args, "seed", None)
    if base_seed is None:
        return None
    return int(base_seed) + max(int(fold_index) - 1, 0)


def train_one_fold(fold, full_data, device, args, fold_index=1):
    daily = full_data["daily_samples"]
    all_dates = full_data["valid_dates"]
    train_dates = [d for d in get_dates_in_range(all_dates, fold["train_start"], fold["train_end"]) if d in daily]
    val_dates = [d for d in get_dates_in_range(all_dates, fold["val_start"], fold["val_end"]) if d in daily]
    test_dates = [d for d in get_dates_in_range(all_dates, fold["test_start"], fold["test_end"]) if d in daily]
    print(f"  Train: {len(train_dates)} | Val: {len(val_dates)} | Test: {len(test_dates)} days")

    if len(train_dates) < 50 or len(val_dates) < 10 or len(test_dates) < 10:
        print("  Skip")
        return pd.DataFrame(), 0.0, 0.0, None, None

    fold_seed = _resolve_fold_seed(args, fold_index)
    if fold_seed is not None:
        set_random_seed(fold_seed, deterministic=getattr(args, "deterministic", False))
        print(f"  Fold seed: {fold_seed} | deterministic={bool(getattr(args, 'deterministic', False))}")
    shuffle_rng = np.random.default_rng(fold_seed)

    max_stocks = max(
        max(len(daily[d]["tickers"]) for d in train_dates),
        max(len(daily[d]["tickers"]) for d in val_dates),
        max(len(daily[d]["tickers"]) for d in test_dates),
    )

    precomputed = {}
    for dt in train_dates:
        precomputed[dt] = _pad_day_sample(daily[dt], max_stocks)
    print(f"  Precomputed {len(precomputed)} days (stored on cpu, batch-loaded to {device})")

    nf = full_data["num_factors"]
    model = QuantV4(
        nf=nf,
        d=args.d_model,
        nh=args.nhead,
        nl=args.num_layers,
        ff=args.d_model * 2,
        drop=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.num_epochs // 2, 1))
    criterion = V4CombinedLoss(
        listnet_weight=args.listnet_weight,
        ic_weight=1.0 - args.listnet_weight,
        temperature=args.temperature,
    )
    amp_enabled = getattr(args, "amp_enabled", False) and device.type == "cuda"
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "float16") == "float16" else torch.bfloat16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    best_val_ic, best_state, no_improve, best_epoch = -np.inf, None, 0, None

    for ep in range(args.num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss, epoch_ic, n_batch = 0.0, 0.0, 0
        shuffled = shuffle_rng.permutation(train_dates).tolist()

        for batch_start in range(0, len(shuffled), args.batch_days):
            batch_dates = shuffled[batch_start:batch_start + args.batch_days]
            x_cpu, y_cpu, mask_cpu = _stack_batch([precomputed[dt] for dt in batch_dates])
            x_batch = _move_to_device(x_cpu, device)
            y_batch = _move_to_device(y_cpu, device)
            mask_batch = _move_to_device(mask_cpu, device)

            optimizer.zero_grad()
            with _amp_context(amp_enabled, amp_dtype):
                pred_batch = model(x_batch, stock_mask=mask_batch)
            pred_batch = pred_batch.float()
            total_loss, total_ic = _batch_loss_and_ic(pred_batch, y_batch, mask_batch, criterion)

            if scaler_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ic += total_ic
            n_batch += 1

            del x_batch, y_batch, mask_batch, pred_batch, total_loss, x_cpu, y_cpu, mask_cpu

        scheduler.step()

        model.eval()
        val_ics = []
        with torch.no_grad():
            val_predictions = _predict_by_date(
                model=model,
                daily=daily,
                dates=val_dates,
                max_stocks=max_stocks,
                device=device,
                batch_days=args.batch_days,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            for dt in val_dates:
                sample = daily[dt]
                ic, _ = stats.spearmanr(val_predictions[dt], sample["y"])
                if not np.isnan(ic):
                    val_ics.append(ic)
        val_ic = np.mean(val_ics) if val_ics else 0.0

        if val_ic > best_val_ic + args.early_stop_min_delta:
            best_val_ic = val_ic
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 10 == 0:
            print(
                f"    Ep {ep + 1:3d} | Loss:{epoch_loss / max(n_batch, 1):.4f} "
                f"| Train IC:{epoch_ic / max(n_batch, 1):.4f} "
                f"| Val IC:{val_ic:.4f} | Best:{best_val_ic:.4f} "
                f"| P:{no_improve}/{args.patience} | Sec:{time.perf_counter() - epoch_start:.1f}"
            )

        if no_improve >= args.patience and ep >= 20:
            print(f"    Early stop at epoch {ep + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    results = []
    with torch.no_grad():
        test_predictions = _predict_by_date(
            model=model,
            daily=daily,
            dates=test_dates,
            max_stocks=max_stocks,
            device=device,
            batch_days=args.batch_days,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for dt in test_dates:
            sample = daily[dt]
            pred_test = test_predictions[dt]
            raw_actuals = sample.get("raw_y", sample["y"])
            for i, ticker in enumerate(sample["tickers"]):
                results.append(
                    {
                        "ticker": ticker,
                        "date": dt,
                        "predicted": pred_test[i],
                        "actual": sample["y"][i],
                        "raw_actual": raw_actuals[i],
                    }
                )

    result_df = pd.DataFrame(results)
    test_ic = stats.spearmanr(result_df["predicted"], result_df["actual"])[0] if len(result_df) > 0 else 0.0
    best_epoch_text = best_epoch if best_epoch is not None else "n/a"
    print(f"  Done | Test IC: {test_ic:.4f} | Best Val IC: {best_val_ic:.4f} | Best Epoch: {best_epoch_text}\n")

    del precomputed, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result_df, best_val_ic, test_ic, fold_seed, best_epoch
