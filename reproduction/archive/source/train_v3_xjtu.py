"""End-to-end XJTU-SY bearing RUL training with PHR-routed config.

Loads from the XJTU-SY mirror provided by the user at
ood_validation/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/.

15 bearings across 3 operating conditions (35Hz12kN, 37.5Hz11kN, 40Hz10kN).
Each bearing has ~123 snapshot CSVs at 25.6 kHz / 1.28 s.

PHR-routed config will be filled in once compute_phr_features.py --dataset XJTU-SY
finishes; for now we use the same template as FEMTO with PHR-derived values
overridden via CLI.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from baselines.models_v3 import DualStreamTCN_v3
from ood_validation.scripts.compute_phr_features import load_xjtu

XJTU_ROOT = Path("ood_validation/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets")
# Defaults will be loaded from ood_validation/XJTU-SY_phr.json if available
SEQ_LEN_DEFAULT = 80           # will be overridden from PHR json
MAX_RUL_DEFAULT = 200.0
LR_DEFAULT = 5e-4
HUBER_DELTA = 1.0
EPOCHS = 60
BATCH = 64
N_CHANNELS = 8   # 4 features x 2 vibration channels


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trend(x: np.ndarray, win: int = 11) -> np.ndarray:
    if x.shape[0] < win:
        return x.copy()
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    smooth = (cumsum[win:] - cumsum[:-win]) / win
    pad = np.repeat(smooth[:1], win, axis=0)
    return np.concatenate([pad, smooth], axis=0)[:x.shape[0]]


class XJTUDataset(Dataset):
    def __init__(self, units, seq_len: int, max_rul: float,
                 mean=None, std=None):
        self.x = []; self.y = []; self.y_unc = []
        for u in units:
            T = u.shape[0]
            if T < seq_len + 1:
                continue
            trend = build_trend(u)
            ruls_unc = np.arange(T - 1, -1, -1, dtype=np.float32)   # uncapped remaining RUL
            ruls = np.minimum(ruls_unc, max_rul)                    # capped target (training)
            for t in range(seq_len, T):
                self.x.append(np.concatenate(
                    [u[t-seq_len:t], trend[t-seq_len:t]], axis=1))
                self.y.append(ruls[t - 1])
                self.y_unc.append(ruls_unc[t - 1])
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        self.y_unc = np.array(self.y_unc, dtype=np.float32)
        if mean is None:
            self.mean = self.x.mean(axis=(0, 1), keepdims=True) if self.x.size else 0
            self.std = self.x.std(axis=(0, 1), keepdims=True) + 1e-6 if self.x.size else 1
        else:
            self.mean = mean; self.std = std
        if self.x.size:
            self.x = (self.x - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]).float(),
                torch.tensor(self.y[idx]).float())


def train_one_seed(units_tr, units_va, units_te, seed, cfg, device):
    set_seed(seed)
    ds_tr = XJTUDataset(units_tr, cfg["seq_len"], cfg["max_rul"])
    ds_va = XJTUDataset(units_va, cfg["seq_len"], cfg["max_rul"],
                         mean=ds_tr.mean, std=ds_tr.std)
    ds_te = XJTUDataset(units_te, cfg["seq_len"], cfg["max_rul"],
                         mean=ds_tr.mean, std=ds_tr.std)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch"], shuffle=True,
                       drop_last=False, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch"], shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch"], shuffle=False, num_workers=0)
    print(f"  seed {seed}: train w={len(ds_tr)}, val w={len(ds_va)}, test w={len(ds_te)}",
          flush=True)
    model = DualStreamTCN_v3(
        num_sensors=N_CHANNELS, d_model=128, variant='V_A',
        disable_patch=False, disable_xstream=False,
        disable_pe=True, disable_trans=True, disable_skfusion=True,
        disable_het=True, disable_length=True,
    ).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["epochs"])
    huber = nn.HuberLoss(delta=cfg["huber_delta"], reduction="mean")

    def eval_loader(dl):
        model.eval(); ps = []
        with torch.no_grad():
            for x, _y in dl:
                x = x.to(device)
                mu, _ = model(x)
                ps.append(mu.reshape(-1).cpu().numpy())
        return np.concatenate(ps)

    # Uncapped/capped test truth, aligned with dl_te order (shuffle=False -> matches ds_te).
    te_cap = ds_te.y.copy(); te_unc = ds_te.y_unc.copy()
    va_y = ds_va.y.copy()
    best_val = float("inf"); best_test_preds = None
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in dl_tr:
            x = x.to(device); y = y.to(device)
            optimiser.zero_grad(set_to_none=True)
            mu, _ = model(x)
            loss = huber(mu.reshape(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()
        val_preds = eval_loader(dl_va)
        val_rmse = float(np.sqrt(((val_preds - va_y) ** 2).mean()))
        # Model selection on VALIDATION (never the test set) — fixes the OOD test leakage.
        if val_rmse < best_val:
            best_val = val_rmse
            best_test_preds = eval_loader(dl_te)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  val_rmse={val_rmse:.3f}  best_val={best_val:.3f}",
                  flush=True)
    return best_val, best_test_preds, te_cap, te_unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--max_rul", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--huber_delta", type=float, default=HUBER_DELTA)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--tag", type=str, default="phr_routed")
    args = ap.parse_args()

    # Load PHR-routed values from compute_phr_features output
    phr_json = Path("ood_validation/XJTU-SY_phr.json")
    seq_len = args.seq_len
    max_rul = args.max_rul
    lr = args.lr
    if phr_json.exists():
        d = json.load(phr_json.open())
        if seq_len is None:
            seq_len = d["routed_hp"]["SEQ_LEN"]
        if max_rul is None:
            max_rul = float(d["routed_hp"]["MAX_RUL"])
        if lr is None:
            lr = float(d["routed_hp"]["LR"])
        print(f"Loaded PHR config from {phr_json}: SEQ={seq_len}, MAX={max_rul}, LR={lr}",
              flush=True)
    if seq_len is None:
        seq_len = SEQ_LEN_DEFAULT
    if max_rul is None:
        max_rul = MAX_RUL_DEFAULT
    if lr is None:
        lr = LR_DEFAULT

    cfg = dict(seq_len=seq_len, max_rul=max_rul, lr=lr,
               epochs=args.epochs, huber_delta=args.huber_delta, batch=args.batch)
    print("=" * 60)
    print("XJTU-SY end-to-end | PHR-routed config")
    print(f"  SEQ_LEN={cfg['seq_len']}  MAX_RUL={cfg['max_rul']}  LR={cfg['lr']}")
    print("=" * 60, flush=True)

    import pickle
    _cache = Path("xjtu_units_cache.pkl")
    if _cache.exists():
        units, meta = pickle.load(_cache.open("rb"))
        print(f"Loaded {len(units)} bearings from cache {_cache}", flush=True)
    else:
        units, meta = load_xjtu(XJTU_ROOT)
        pickle.dump((units, meta), _cache.open("wb"))
        print(f"Built + cached {len(units)} bearings -> {_cache}", flush=True)
    print(f"  avg len {np.mean([u.shape[0] for u in units]):.0f}, "
          f"lengths {sorted(int(u.shape[0]) for u in units)}", flush=True)
    # Split: one bearing per condition held out for TEST and one for VAL. Best epoch is
    # selected on VAL (never TEST) -> no test leakage. Groups of 5 per condition (idx 0..14).
    test_idx = [4, 9, 14]
    val_idx  = [3, 8, 13]
    units_tr = [u for i, u in enumerate(units) if i not in test_idx and i not in val_idx]
    units_va = [u for i, u in enumerate(units) if i in val_idx]
    units_te = [u for i, u in enumerate(units) if i in test_idx]
    print(f"  train/val/test bearings: {len(units_tr)}/{len(units_va)}/{len(units_te)}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    per_seed_val = []; all_preds = []; trues_cap = None; trues_unc = None
    for seed in [42, 43, 44, 45, 46][:args.seeds]:
        vrmse, preds, tcap, tunc = train_one_seed(units_tr, units_va, units_te, seed, cfg, device)
        per_seed_val.append(vrmse); all_preds.append(preds)
        if trues_cap is None:
            trues_cap = tcap; trues_unc = tunc
    median_pred = np.median(np.stack(all_preds, axis=0), axis=0)
    ens_cap = np.clip(median_pred, 0, cfg["max_rul"])
    ensemble_rmse = float(np.sqrt(((ens_cap - trues_cap) ** 2).mean()))

    out_path = Path("artifacts_v3") / f"XJTU-SY_V_A_{args.tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    json.dump({
        "dataset": "XJTU-SY",
        "tag": args.tag,
        "config": cfg,
        "n_train_traj": len(units_tr),
        "n_val_traj": len(units_va),
        "n_test_traj": len(units_te),
        "per_seed_val_rmse": per_seed_val,
        "ensemble_rmse": ensemble_rmse,
        "ensemble_unit_preds": [float(x) for x in median_pred],
        "test_true_capped": [float(x) for x in trues_cap],
        "test_true_uncapped": [float(x) for x in trues_unc],
        "train_cap": float(cfg["max_rul"]),
        "phr_source": str(phr_json),
    }, out_path.open("w"), indent=2)
    print()
    print(f"XJTU-SY DONE  per_seed_val={[f'{r:.3f}' for r in per_seed_val]}  "
          f"ensemble RMSE(capped)={ensemble_rmse:.3f}  n_test_windows={len(trues_cap)}", flush=True)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
