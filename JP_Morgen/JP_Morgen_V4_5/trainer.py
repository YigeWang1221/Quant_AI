import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dateutil.relativedelta import relativedelta
from scipy import stats

from loss import V4CombinedLoss
from model import QuantV4


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


def train_one_fold_v4(fold, full_data, device, args):
    daily = full_data["daily_samples"]
    all_dates = full_data["valid_dates"]
    train_dates = [d for d in get_dates_in_range(all_dates, fold["train_start"], fold["train_end"]) if d in daily]
    val_dates = [d for d in get_dates_in_range(all_dates, fold["val_start"], fold["val_end"]) if d in daily]
    test_dates = [d for d in get_dates_in_range(all_dates, fold["test_start"], fold["test_end"]) if d in daily]
    print(f"  Train: {len(train_dates)} | Val: {len(val_dates)} | Test: {len(test_dates)} days")

    if len(train_dates) < 50 or len(val_dates) < 10 or len(test_dates) < 10:
        print("  Skip")
        return pd.DataFrame(), 0.0, 0.0

    max_stocks = max(
        max(len(daily[d]["tickers"]) for d in train_dates),
        max(len(daily[d]["tickers"]) for d in val_dates),
        max(len(daily[d]["tickers"]) for d in test_dates),
    )

    precomputed = {}
    for dt in train_dates:
        sample = daily[dt]
        x = torch.FloatTensor(sample["X"])
        y = torch.FloatTensor(sample["y"])
        n_stocks = x.shape[0]
        if n_stocks < max_stocks:
            x = torch.cat([x, torch.zeros(max_stocks - n_stocks, x.shape[1], x.shape[2])])
            y = torch.cat([y, torch.zeros(max_stocks - n_stocks)])
        mask = torch.zeros(max_stocks)
        mask[:n_stocks] = 1.0
        precomputed[dt] = (x.to(device), y.to(device), mask.to(device))
    print(f"  Precomputed {len(precomputed)} days (on {device})")

    nf = full_data["num_factors"]
    model = QuantV4(
        nf=nf,
        d=args.d_model,
        nh=args.nhead,
        nl=args.num_layers,
        ff=args.d_model * 2,
        drop=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.num_epochs // 2, 1))
    criterion = V4CombinedLoss(
        listnet_weight=args.listnet_weight,
        ic_weight=1.0 - args.listnet_weight,
        temperature=args.temperature,
    )

    best_val_ic, best_state, no_improve = -np.inf, None, 0

    for ep in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_ic, n_batch = 0.0, 0.0, 0
        shuffled = train_dates.copy()
        np.random.shuffle(shuffled)
        batches = []

        for batch_start in range(0, len(shuffled), args.batch_days):
            batch_dates = shuffled[batch_start:batch_start + args.batch_days]
            xs, ys, masks = zip(*[precomputed[dt] for dt in batch_dates])
            batches.append((torch.stack(xs), torch.stack(ys), torch.stack(masks)))

        for x_batch, y_batch, mask_batch in batches:
            optimizer.zero_grad()
            batch_size = x_batch.shape[0]
            preds = [model(x_batch[i], stock_mask=mask_batch[i]) for i in range(batch_size)]
            pred_batch = torch.stack(preds)

            total_loss = torch.tensor(0.0, device=device)
            total_ic = 0.0
            for i in range(batch_size):
                loss_i, _, ic_i = criterion(pred_batch[i], y_batch[i], mask_batch[i])
                total_loss = total_loss + loss_i
                total_ic += ic_i
            total_loss = total_loss / batch_size

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ic += total_ic / batch_size
            n_batch += 1

        scheduler.step()

        model.eval()
        val_ics = []
        with torch.no_grad():
            for dt in val_dates:
                sample = daily[dt]
                x_val = torch.FloatTensor(sample["X"]).to(device)
                pred_val = model(x_val).cpu().numpy()
                ic, _ = stats.spearmanr(pred_val, sample["y"])
                if not np.isnan(ic):
                    val_ics.append(ic)
        val_ic = np.mean(val_ics) if val_ics else 0.0

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 10 == 0:
            print(
                f"    Ep {ep + 1:3d} | Loss:{epoch_loss / max(n_batch, 1):.4f} "
                f"| Train IC:{epoch_ic / max(n_batch, 1):.4f} "
                f"| Val IC:{val_ic:.4f} | Best:{best_val_ic:.4f} "
                f"| P:{no_improve}/{args.patience}"
            )

        if no_improve >= args.patience and ep >= 20:
            print(f"    Early stop at epoch {ep + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    results = []
    with torch.no_grad():
        for dt in test_dates:
            sample = daily[dt]
            x_test = torch.FloatTensor(sample["X"]).to(device)
            pred_test = model(x_test).cpu().numpy()
            for i, ticker in enumerate(sample["tickers"]):
                results.append({"ticker": ticker, "date": dt, "predicted": pred_test[i], "actual": sample["y"][i]})

    result_df = pd.DataFrame(results)
    test_ic = stats.spearmanr(result_df["predicted"], result_df["actual"])[0] if len(result_df) > 0 else 0.0
    print(f"  Done | Test IC: {test_ic:.4f} | Best Val IC: {best_val_ic:.4f}\n")

    del precomputed, batches, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result_df, best_val_ic, test_ic
