"""
train_v3_regression.py
======================
Unified regression trainer for CMAPSS FD001-FD004 and NASA Battery.

Each dataset reuses its own preprocessed pickle (already produced by the
existing per-experiment preprocess.py).  The training pipeline is identical:

  * DualStreamTCN_v3 backbone, variant ∈ {V_A, V_Aprime}, with ablation flags
  * Heteroscedastic head (μ, logσ²) trained with Gaussian NLL
  * Length-token feature feeding into the head
  * Median-of-σ⁻²-weighted ensemble across N seeds (Bayesian-style fusion)
  * Saves best ckpt per (dataset, variant, ablation, seed) under
      checkpoints_v3/<dataset>/<variant>_<tag>_seed<N>.pth

CLI:
    python train_v3_regression.py --dataset FD001 --variant V_A --seeds 3
    python train_v3_regression.py --dataset FD002 --variant V_Aprime --disable_xstream

Results dump:
    artifacts_v3/<dataset>_<variant>_<tag>.json
        { 'rmse': float, 'score': float, 'per_seed': [...], ... }
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from baselines.models_v3 import DualStreamTCN_v3, gaussian_nll_loss

# The FD002/FD004 meta.pkl contains pickled `ClusteredNormalizer` instances.
# Python's unpickler looks up the class in the caller's module (`__main__`), so we
# need to re-import / re-define it here. We borrow it from the FD002 preprocessor.
sys.path.insert(0, str(Path(__file__).parent / 'Experiment_FD002'))
try:
    from preprocess import ClusteredNormalizer  # noqa: F401
except Exception:
    # Fallback: minimal stub that still loads (we never refit, only unpickle to get raw_cols/trend_cols).
    class ClusteredNormalizer:  # noqa
        pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Per-dataset configuration
DATASETS = {
    'FD001': dict(
        proc_dir='Experiment_FD001/processed',
        subset='FD001', seq_len=50, max_rul=130.0, lr=5e-4, epochs=150, patience=25,
        batch=128, online_trend=True, trend_window=11,
    ),
    'FD002': dict(
        proc_dir='Experiment_FD002/processed',
        subset='FD002', seq_len=50, max_rul=130.0, lr=1e-3, epochs=120, patience=20,
        batch=128, online_trend=False,
    ),
    'FD003': dict(
        proc_dir='Experiment_FD003/processed',
        subset='FD003', seq_len=80, max_rul=125.0, lr=1e-3, epochs=120, patience=25,
        batch=128, online_trend=True, trend_window=11,
    ),
    'FD004': dict(
        proc_dir='Experiment_FD004/processed',
        subset='FD004', seq_len=50, max_rul=130.0, lr=5e-4, epochs=150, patience=25,
        batch=128, online_trend=False,
    ),
    'Battery': dict(
        proc_dir='Experiment_Battery/processed_battery_new',
        subset='Battery', seq_len=70, max_rul=110.0, lr=5e-4, epochs=120, patience=25,
        batch=64, online_trend='battery',
    ),
}


# Dataset wrappers
class CMAPSSDataset(Dataset):
    """Builds sliding windows for FD001-4, computing trend online when needed.
    Each item: (x [seq_len, 2*ns] float32, y float32, slen float32 normalised).
    """
    def __init__(self, df, raw_cols, trend_cols, seq_len, mode, online_trend,
                 trend_window=11, max_rul=130.0, augment=False, external_rul=None,
                 aug_truncate_prob=0.30, aug_truncate_min_frac=0.4,
                 aug_noise_std=0.02, aug_scale_range=(0.95, 1.05)):
        self.seq_len = seq_len
        self.mode = mode
        self.augment = augment and mode == 'train'
        self.aug_truncate_prob = aug_truncate_prob
        self.aug_truncate_min_frac = aug_truncate_min_frac
        self.aug_noise_std = aug_noise_std
        self.aug_scale_range = aug_scale_range

        self.x_data, self.y_data, self.slen_data = [], [], []

        units = sorted(df['unit'].unique())
        for uid in units:
            g = df[df['unit'] == uid].sort_values('cycle') if 'cycle' in df.columns else df[df['unit'] == uid]
            raw = g[raw_cols].values.astype(np.float32)
            if online_trend:
                trend_df = pd.DataFrame(raw)
                trend = trend_df.rolling(window=trend_window, min_periods=1).mean().values.astype(np.float32)
            else:
                trend = g[trend_cols].values.astype(np.float32)

            if mode == 'test':
                # External RUL is the ground-truth at the LAST cycle of the test sequence.
                rul = np.full(len(raw), external_rul[uid - 1] if external_rul is not None else 0.0,
                              dtype=np.float32)
            else:
                rul = np.clip(g['RUL'].values.astype(np.float32), 0, max_rul) if mode == 'train' \
                      else g['RUL'].values.astype(np.float32)

            n = max(1, len(raw) - seq_len + 1)
            for i in range(n):
                end = i + seq_len
                if end > len(raw):
                    pad = end - len(raw)
                    raw_seq = np.concatenate([np.tile(raw[0], (pad, 1)), raw[i:]], axis=0)
                    trend_seq = np.concatenate([np.tile(trend[0], (pad, 1)), trend[i:]], axis=0)
                    eff_len = len(raw) - i  # how many real (non-pad) cycles inside this window
                    y_val = rul[-1]
                else:
                    raw_seq = raw[i:end]
                    trend_seq = trend[i:end]
                    eff_len = seq_len
                    y_val = rul[end - 1]
                self.x_data.append(np.concatenate([raw_seq, trend_seq], axis=1))
                self.y_data.append(y_val)
                # Length feature: log(eff_len) / log(seq_len) ∈ (0, 1]
                self.slen_data.append(math.log(max(eff_len, 1)) / math.log(seq_len))

        self.x_data = np.array(self.x_data, dtype=np.float32)
        self.y_data = np.array(self.y_data, dtype=np.float32)
        self.slen_data = np.array(self.slen_data, dtype=np.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]; y = self.y_data[idx]; slen = self.slen_data[idx]
        if self.augment:
            x = x.copy()
            half = x.shape[1] // 2
            if np.random.rand() > 0.5:
                x[:, :half] += np.random.normal(0, self.aug_noise_std, x[:, :half].shape).astype(np.float32)
            if np.random.rand() > 0.5:
                lo, hi = self.aug_scale_range
                x[:, :half] *= (lo + (hi - lo) * np.random.rand())
            if np.random.rand() < self.aug_truncate_prob:
                T = x.shape[0]
                keep = max(8, int(T * (self.aug_truncate_min_frac
                                        + (1 - self.aug_truncate_min_frac) * np.random.rand())))
                pad = T - keep
                front = np.tile(x[0:1], (pad, 1))
                x = np.concatenate([front, x[T - keep:]], axis=0)
                slen = math.log(keep) / math.log(T)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(slen, dtype=torch.float32))


# Loaders
def load_cmapss(cfg):
    proc = Path(cfg['proc_dir'])
    train_df = pd.read_pickle(proc / f"train_{cfg['subset']}.pkl")
    test_df  = pd.read_pickle(proc / f"test_{cfg['subset']}.pkl")
    truth_df = pd.read_pickle(proc / f"truth_{cfg['subset']}.pkl")
    with open(proc / f"meta_{cfg['subset']}.pkl", 'rb') as f:
        meta = pickle.load(f)
    return train_df, test_df, truth_df, meta


def load_battery(cfg):
    """Battery uses processed_battery_new dir with cell-level pickles."""
    proc = Path(cfg['proc_dir'])
    # Battery pipeline saves a single combined frame
    if (proc / 'train_battery.pkl').exists():
        train_df = pd.read_pickle(proc / 'train_battery.pkl')
        test_df  = pd.read_pickle(proc / 'test_battery.pkl')
        truth_df = pd.read_pickle(proc / 'truth_battery.pkl')
        with open(proc / 'meta_battery.pkl', 'rb') as f:
            meta = pickle.load(f)
        return train_df, test_df, truth_df, meta
    raise FileNotFoundError(f"Battery processed files not found in {proc}")


def make_loaders(cfg, train_df, test_df, truth_df, meta, seed):
    raw_cols = meta['raw_cols']
    trend_cols = meta.get('trend_cols', None)
    online_trend = bool(cfg['online_trend'])  # truthy → online; "battery" still truthy

    # 85/15 unit split with per-seed shuffle for diversity
    units = sorted(train_df['unit'].unique())
    rng = np.random.default_rng(seed)
    units = rng.permutation(units)
    n_val = max(1, int(len(units) * 0.15))
    val_units = units[:n_val]; train_units = units[n_val:]
    df_tr = train_df[train_df['unit'].isin(train_units)].reset_index(drop=True)
    df_va = train_df[train_df['unit'].isin(val_units)].reset_index(drop=True)

    ds_tr = CMAPSSDataset(df_tr, raw_cols, trend_cols, cfg['seq_len'], 'train',
                          online_trend, cfg.get('trend_window', 11), cfg['max_rul'], augment=True)
    ds_va = CMAPSSDataset(df_va, raw_cols, trend_cols, cfg['seq_len'], 'train',
                          online_trend, cfg.get('trend_window', 11), cfg['max_rul'], augment=False)
    external_rul = truth_df['RUL'].values.astype(np.float32) if 'RUL' in truth_df.columns else None
    ds_te = CMAPSSDataset(test_df, raw_cols, trend_cols, cfg['seq_len'], 'test',
                          online_trend, cfg.get('trend_window', 11), cfg['max_rul'],
                          augment=False, external_rul=external_rul)

    dl_tr = DataLoader(ds_tr, cfg['batch'], shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    dl_va = DataLoader(ds_va, cfg['batch'] * 2, shuffle=False, pin_memory=True, num_workers=0)
    dl_te = DataLoader(ds_te, cfg['batch'] * 2, shuffle=False, pin_memory=True, num_workers=0)
    return dl_tr, dl_va, dl_te, ds_te, raw_cols


# Metrics
def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate_test_with_sigma(model, dl_te, test_df, truth_df, cfg, het=True, ema_span=3):
    """Slide windows, ema-aggregate per-unit, return (preds, sigmas, trues)."""
    model.eval()
    all_mu, all_var = [], []
    with torch.no_grad():
        for x, _, slen in dl_te:
            x = x.to(DEVICE); slen = slen.to(DEVICE).unsqueeze(-1)
            with autocast():
                mu, logvar = model(x, slen)
            all_mu.append(mu.float().cpu().numpy())
            if het and logvar is not None:
                all_var.append(torch.exp(logvar).float().cpu().numpy())
            else:
                all_var.append(np.ones_like(mu.float().cpu().numpy()))
    mu_arr = np.concatenate(all_mu)
    var_arr = np.concatenate(all_var)

    unit_preds, unit_sig, idx = [], [], 0
    sorted_groups = sorted(test_df.groupby('unit'), key=lambda x: x[0])
    for _, g in sorted_groups:
        n = max(1, len(g) - cfg['seq_len'] + 1)
        traj_mu = mu_arr[idx: idx + n]
        traj_var = var_arr[idx: idx + n]
        if len(traj_mu) >= 2:
            smoothed = pd.Series(traj_mu).ewm(span=ema_span, adjust=False).mean().values
            unit_preds.append(float(smoothed[-1]))
            unit_sig.append(float(traj_var[-1]))
        else:
            unit_preds.append(float(traj_mu[-1]))
            unit_sig.append(float(traj_var[-1]))
        idx += n
    unit_preds = np.clip(np.array(unit_preds), 0, cfg['max_rul'])
    trues = np.clip(truth_df['RUL'].values, 0, cfg['max_rul'])
    return unit_preds, np.array(unit_sig), trues


# Single-seed training
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_seed(cfg, dataset_name, raw_cols, trend_cols, dl_tr, dl_va, dl_te, ds_te,
                    test_df, truth_df, args, seed, save_path):
    set_seed(seed)
    num_sensors = len(raw_cols)

    model = DualStreamTCN_v3(
        num_sensors=num_sensors,
        d_model=128,
        variant=args.variant,
        disable_patch=args.disable_patch,
        disable_xstream=args.disable_xstream,
        disable_pe=args.disable_pe,
        disable_trans=args.disable_trans,
        disable_skfusion=args.disable_skfusion,
        disable_het=args.disable_het,
        disable_length=args.disable_length,
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-6)
    scaler = GradScaler()
    huber = nn.HuberLoss(delta=args.huber_delta)

    best_score = float('inf'); best_rmse = float('inf')
    patience_counter = 0
    nan_streak = 0
    swa_model = None
    swa_start_ep = int(cfg['epochs'] * args.swa_start_ratio) if args.use_swa else 10 ** 9
    for ep in range(cfg['epochs']):
        model.train(); tr_loss = 0.0
        for x, y, slen in dl_tr:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            slen = slen.to(DEVICE, non_blocking=True).unsqueeze(-1)
            opt.zero_grad(set_to_none=True)
            with autocast():
                mu, logvar = model(x, slen)
                if (not args.disable_het) and logvar is not None:
                    loss = gaussian_nll_loss(mu, logvar, y)
                else:
                    loss = huber(mu, y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()
        sched.step()

        # Validation
        model.eval(); val_mu, val_y = [], []
        with torch.no_grad():
            for x, y, slen in dl_va:
                x = x.to(DEVICE); slen = slen.to(DEVICE).unsqueeze(-1)
                with autocast():
                    mu, logvar = model(x, slen)
                val_mu.append(mu.float().cpu().numpy()); val_y.append(y.numpy())
        val_mu = np.concatenate(val_mu); val_y = np.concatenate(val_y)
        val_mu = np.clip(np.nan_to_num(val_mu, nan=cfg['max_rul']/2.0), 0, cfg['max_rul'])
        val_y = np.clip(val_y, 0, cfg['max_rul'])
        rmse = float(np.sqrt(np.mean((val_mu - val_y) ** 2)))
        score = nasa_score(val_mu, val_y)
        if not np.isfinite(rmse):
            nan_streak += 1
            if nan_streak >= 3:
                print(f"  Aborting seed {seed} after 3 NaN epochs"); break
            continue
        nan_streak = 0

        if score < best_score:
            best_score = score; best_rmse = rmse; patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"  [es] seed {seed} stop @ epoch {ep+1}  best RMSE={best_rmse:.3f}/{best_score:.1f}")
                break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:03d} tr_loss={tr_loss/max(len(dl_tr),1):.3f}  val_rmse={rmse:.3f}/{score:.0f}"
                  f"  best={best_rmse:.3f}/{best_score:.0f}")

        # SWA collection in the last (1 - swa_start_ratio) fraction of epochs
        if args.use_swa and ep >= swa_start_ep:
            if swa_model is None:
                from torch.optim.swa_utils import AveragedModel
                swa_model = AveragedModel(model)
            else:
                swa_model.update_parameters(model)

    # SWA finalisation
    if args.use_swa and swa_model is not None:
        from torch.optim.swa_utils import update_bn
        swa_model = swa_model.to(DEVICE)
        try:
            update_bn(dl_tr, swa_model, device=DEVICE)
        except Exception as e:
            print(f"  [SWA] update_bn skipped: {e}")
        # Eval SWA on val and compare to best
        swa_model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for x, y, slen in dl_va:
                x = x.to(DEVICE); slen = slen.to(DEVICE).unsqueeze(-1)
                out = swa_model(x, slen)
                mu = out[0] if isinstance(out, tuple) else out
                vp.append(mu.float().cpu().numpy()); vy.append(y.numpy())
        vp = np.clip(np.nan_to_num(np.concatenate(vp), nan=cfg['max_rul']/2), 0, cfg['max_rul'])
        vy = np.clip(np.concatenate(vy), 0, cfg['max_rul'])
        swa_rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
        swa_score = nasa_score(vp, vy)
        if swa_score < best_score:
            print(f"  [SWA] WIN rmse={swa_rmse:.3f} score={swa_score:.1f} → overwriting best ckpt")
            torch.save(swa_model.module.state_dict(), save_path)
            best_rmse, best_score = swa_rmse, swa_score
        else:
            print(f"  [SWA] keep regular  swa={swa_rmse:.3f}/{swa_score:.1f}  best={best_rmse:.3f}/{best_score:.1f}")
        del swa_model

    if not os.path.exists(save_path):
        print(f"  seed {seed} produced no checkpoint, skipping"); return None
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    preds, sigmas, trues = evaluate_test_with_sigma(
        model, dl_te, test_df, truth_df, cfg,
        het=not args.disable_het, ema_span=args.ema_span)
    return dict(preds=preds, sigmas=sigmas, trues=trues, best_rmse=best_rmse, best_score=best_score)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    ap.add_argument('--variant', default='V_A', choices=['V_A', 'V_Aprime'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed_base', type=int, default=42)
    ap.add_argument('--tag', default='')  # for ablation naming
    ap.add_argument('--disable_patch', action='store_true')
    ap.add_argument('--disable_xstream', action='store_true')
    ap.add_argument('--disable_pe', action='store_true')
    ap.add_argument('--disable_trans', action='store_true')
    ap.add_argument('--disable_skfusion', action='store_true')
    ap.add_argument('--disable_het', action='store_true')
    ap.add_argument('--disable_length', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--use_swa', action='store_true', help='Average weights over last 40% of epochs')
    ap.add_argument('--swa_start_ratio', type=float, default=0.6)
    # Hyperparameter overrides (no architectural change — keep proposed model intact)
    ap.add_argument('--seq_len',     type=int,   default=None, help='Override window length')
    ap.add_argument('--max_rul',     type=float, default=None, help='Override RUL ceiling')
    ap.add_argument('--lr',          type=float, default=None, help='Override learning rate')
    ap.add_argument('--batch',       type=int,   default=None, help='Override batch size')
    ap.add_argument('--patience',    type=int,   default=None, help='Override early-stop patience')
    ap.add_argument('--huber_delta', type=float, default=1.0,  help='Huber loss delta')
    ap.add_argument('--ema_span',    type=int,   default=3,    help='EMA window for sliding-pred aggregation')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset].copy()
    if args.epochs is not None:   cfg['epochs']   = args.epochs
    if args.seq_len is not None:  cfg['seq_len']  = args.seq_len
    if args.max_rul is not None:  cfg['max_rul']  = args.max_rul
    if args.lr is not None:       cfg['lr']       = args.lr
    if args.batch is not None:    cfg['batch']    = args.batch
    if args.patience is not None: cfg['patience'] = args.patience

    print(f"\n{'='*70}")
    print(f"DATASET={args.dataset}  VARIANT={args.variant}  TAG={args.tag or 'full'}")
    print(f"  ablation flags: patch={args.disable_patch} xstream={args.disable_xstream} "
          f"pe={args.disable_pe} trans={args.disable_trans} skf={args.disable_skfusion} "
          f"het={args.disable_het} length={args.disable_length}")
    print(f"{'='*70}")

    if args.dataset == 'Battery':
        train_df, test_df, truth_df, meta = load_battery(cfg)
    else:
        train_df, test_df, truth_df, meta = load_cmapss(cfg)

    raw_cols = meta['raw_cols']
    trend_cols = meta.get('trend_cols', None)

    tag = args.tag or 'full'
    out_dir = Path('checkpoints_v3') / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path('artifacts_v3'); art_dir.mkdir(exist_ok=True)

    seeds = [args.seed_base + i for i in range(args.seeds)]
    per_seed = []
    all_pred_mu, all_pred_sig, trues = [], [], None
    for seed in seeds:
        ckpt = out_dir / f"{args.variant}_{tag}_seed{seed}.pth"
        dl_tr, dl_va, dl_te, ds_te, _ = make_loaders(cfg, train_df, test_df, truth_df, meta, seed)
        res = train_one_seed(cfg, args.dataset, raw_cols, trend_cols, dl_tr, dl_va, dl_te, ds_te,
                              test_df, truth_df, args, seed, str(ckpt))
        if res is None:
            continue
        per_seed.append(dict(seed=seed, best_rmse=res['best_rmse'], best_score=res['best_score']))
        all_pred_mu.append(res['preds']); all_pred_sig.append(res['sigmas']); trues = res['trues']

    if not all_pred_mu:
        print("All seeds failed — nothing to ensemble.")
        return

    mu_stack = np.array(all_pred_mu)        # [n_seeds, n_units]
    var_stack = np.array(all_pred_sig)
    # Inverse-variance weighted ensemble fallbacks to mean if het disabled.
    # Use nan-aware reductions so a single diverged seed (all-NaN preds) is DROPPED from the
    # ensemble rather than poisoning every unit's median/mean. Identical to the plain reduction
    # when no seed diverged; only affects runs with a NaN seed (e.g. FD002 LR instability).
    if args.disable_het:
        ens = np.nanmedian(mu_stack, axis=0)
    else:
        w = 1.0 / np.maximum(var_stack, 1e-3)
        w = np.where(np.isfinite(mu_stack), w, 0.0)
        ens = np.nansum(mu_stack * w, axis=0) / np.maximum(np.sum(w, axis=0), 1e-12)
    ens = np.clip(ens, 0, cfg['max_rul'])
    ensemble_rmse = float(np.sqrt(np.mean((ens - trues) ** 2)))
    ensemble_score = nasa_score(ens, trues)

    # Per-seed TEST metrics — enables reporting mean +/- std on the *reported* (test)
    # metric. NOTE: per_seed_rmse in the result dict are VALIDATION RMSEs (used for
    # checkpoint selection); the values below are the test-set per-seed numbers that
    # should back any mean/std reported in the paper.
    per_seed_test_rmse = [float(np.sqrt(np.mean((np.clip(mu, 0, cfg['max_rul']) - trues) ** 2)))
                          for mu in all_pred_mu]
    per_seed_test_score = [nasa_score(np.clip(mu, 0, cfg['max_rul']), trues) for mu in all_pred_mu]
    test_rmse_mean = float(np.nanmean(per_seed_test_rmse))
    test_rmse_std  = float(np.nanstd(per_seed_test_rmse))

    print(f"\n=== ENSEMBLE {args.dataset} {args.variant} {tag} ===")
    print(f"  Per-seed TEST RMSE: mean={test_rmse_mean:.3f} std={test_rmse_std:.3f} "
          f"vals={[round(r, 3) for r in per_seed_test_rmse]}")
    per_seed_str = [f"{p['best_rmse']:.3f}/{p['best_score']:.0f}" for p in per_seed]
    print(f"  Per-seed RMSE/Score: {per_seed_str}")
    print(f"  Ensemble RMSE: {ensemble_rmse:.4f}")
    print(f"  Ensemble Score: {ensemble_score:.2f}")

    result = dict(
        dataset=args.dataset, variant=args.variant, tag=tag,
        seeds=[s['seed'] for s in per_seed],
        per_seed_rmse=[s['best_rmse'] for s in per_seed],
        per_seed_score=[s['best_score'] for s in per_seed],
        ensemble_rmse=ensemble_rmse,
        ensemble_score=ensemble_score,
        per_seed_test_rmse=per_seed_test_rmse,
        per_seed_test_score=per_seed_test_score,
        test_rmse_mean=test_rmse_mean,
        test_rmse_std=test_rmse_std,
        # For the cost-regret RUL-ceiling gate: per-unit ensemble prediction (clipped to the
        # training cap) and the UNCLIPPED true RUL, so maintenance cost can be scored against
        # a fixed uncapped reference instead of the cap-dependent clipped target.
        ensemble_unit_preds=[float(x) for x in ens],
        test_true_uncapped=[float(x) for x in truth_df['RUL'].values],
        train_cap=float(cfg['max_rul']),
        flags=dict(disable_patch=args.disable_patch, disable_xstream=args.disable_xstream,
                   disable_pe=args.disable_pe, disable_trans=args.disable_trans,
                   disable_skfusion=args.disable_skfusion, disable_het=args.disable_het,
                   disable_length=args.disable_length),
    )
    art_path = art_dir / f"{args.dataset}_{args.variant}_{tag}.json"
    with open(art_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → wrote {art_path}")


if __name__ == '__main__':
    main()
