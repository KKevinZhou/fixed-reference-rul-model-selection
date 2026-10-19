"""Leakage-resistant C-MAPSS training for the RUL-cap study.

This script deliberately separates four quantities that were conflated in the
legacy experiments:

1. ``train_cap`` controls the labels used to train a model.
2. ``reporting_cap`` defines one common validation/test yardstick.
3. ``split_seed`` fixes the train/validation units for every model seed.
4. ``model_seed`` controls only stochastic model fitting.

During ``--validation_only`` runs the test set is never evaluated and no test
metric is written.  Hyperparameter optimisation must use that mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baselines.models_v3 import DualStreamTCN_v3, gaussian_nll_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULTS = {
    "FD001": dict(proc_dir="Experiment_FD001/processed", seq_len=50, lr=5e-4),
    "FD002": dict(proc_dir="Experiment_FD002/processed", seq_len=50, lr=1e-3),
    "FD003": dict(proc_dir="Experiment_FD003/processed", seq_len=80, lr=1e-3),
    "FD004": dict(proc_dir="Experiment_FD004/processed", seq_len=50, lr=5e-4),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_original_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Recover uncapped run-to-failure labels from unit cycle indices.

    The distributed preprocessed C-MAPSS frames contain labels clipped at 125.
    Their original labels are exactly ``max_cycle(unit) - cycle``.  The strict
    assertion prevents silently applying this reconstruction to incompatible
    data.
    """
    required = {"unit", "cycle", "RUL"}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"Cannot reconstruct original RUL; missing {sorted(missing)}")

    frame = train_df.copy()
    original = (
        frame.groupby("unit", sort=False)["cycle"].transform("max") - frame["cycle"]
    ).astype(np.float32)
    expected_stored = np.minimum(original.to_numpy(), 125.0)
    stored = frame["RUL"].to_numpy(dtype=np.float32)
    if not np.allclose(stored, expected_stored, atol=1e-6):
        max_error = float(np.max(np.abs(stored - expected_stored)))
        raise ValueError(
            "Processed labels are not the expected min(original_RUL, 125); "
            f"maximum discrepancy={max_error:.6g}"
        )
    frame["RUL_ORIGINAL"] = original
    return frame


def load_data(data_root: Path, dataset: str, validation_only: bool):
    spec = DEFAULTS[dataset]
    proc = data_root / spec["proc_dir"]
    train_path = proc / f"train_{dataset}.pkl"
    meta_path = proc / f"meta_{dataset}.pkl"

    # Some metadata pickles refer to preprocess.ClusteredNormalizer.
    sys.path.insert(0, str(data_root / "Experiment_FD002"))
    try:
        import preprocess  # noqa: F401
    except Exception:
        pass

    train_df = reconstruct_original_rul(pd.read_pickle(train_path))
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)

    test_df = truth_df = None
    test_path = truth_path = None
    if not validation_only:
        test_path = proc / f"test_{dataset}.pkl"
        truth_path = proc / f"truth_{dataset}.pkl"
        test_df = pd.read_pickle(test_path)
        truth_df = pd.read_pickle(truth_path)

    fingerprints = {
        "train_sha256": file_sha256(train_path),
        "meta_sha256": file_sha256(meta_path),
    }
    if test_path is not None and truth_path is not None:
        fingerprints.update(
            test_sha256=file_sha256(test_path),
            truth_sha256=file_sha256(truth_path),
        )
    return train_df, test_df, truth_df, meta, fingerprints


class WindowDataset(Dataset):
    """Sliding-window C-MAPSS data with original-scale labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        raw_cols: list[str],
        trend_cols: list[str] | None,
        seq_len: int,
        online_trend: bool,
        trend_window: int = 11,
        augment: bool = False,
        external_rul: np.ndarray | None = None,
    ):
        self.seq_len = seq_len
        self.augment = augment
        xs, ys, lengths = [], [], []

        for uid in sorted(frame["unit"].unique()):
            group = frame[frame["unit"] == uid].sort_values("cycle")
            raw = group[raw_cols].to_numpy(dtype=np.float32)
            if online_trend:
                trend = (
                    pd.DataFrame(raw)
                    .rolling(window=trend_window, min_periods=1)
                    .mean()
                    .to_numpy(dtype=np.float32)
                )
            else:
                if trend_cols is None:
                    raise ValueError("trend_cols are required when online_trend=False")
                trend = group[trend_cols].to_numpy(dtype=np.float32)

            if external_rul is None:
                targets = group["RUL_ORIGINAL"].to_numpy(dtype=np.float32)
            else:
                targets = np.full(
                    len(raw), external_rul[int(uid) - 1], dtype=np.float32
                )

            n_windows = max(1, len(raw) - seq_len + 1)
            for start in range(n_windows):
                end = start + seq_len
                if end > len(raw):
                    pad = end - len(raw)
                    raw_seq = np.concatenate(
                        [np.repeat(raw[:1], pad, axis=0), raw[start:]], axis=0
                    )
                    trend_seq = np.concatenate(
                        [np.repeat(trend[:1], pad, axis=0), trend[start:]], axis=0
                    )
                    effective_length = len(raw) - start
                    target = targets[-1]
                else:
                    raw_seq = raw[start:end]
                    trend_seq = trend[start:end]
                    effective_length = seq_len
                    target = targets[end - 1]
                xs.append(np.concatenate([raw_seq, trend_seq], axis=1))
                ys.append(target)
                lengths.append(
                    math.log(max(effective_length, 1)) / math.log(seq_len)
                )

        self.x = np.asarray(xs, dtype=np.float32)
        self.y = np.asarray(ys, dtype=np.float32)
        self.length = np.asarray(lengths, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        x = self.x[index]
        length = self.length[index]
        if self.augment:
            x = x.copy()
            half = x.shape[1] // 2
            if np.random.rand() > 0.5:
                x[:, :half] += np.random.normal(
                    0.0, 0.02, x[:, :half].shape
                ).astype(np.float32)
            if np.random.rand() > 0.5:
                x[:, :half] *= np.random.uniform(0.95, 1.05)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[index], dtype=torch.float32),
            torch.tensor(length, dtype=torch.float32),
        )


def split_units(train_df: pd.DataFrame, split_seed: int, val_fraction: float):
    """Deterministic group split stratified by observed trajectory length."""
    lifetimes = (
        train_df.groupby("unit", sort=True)["cycle"].max().sort_index()
    )
    units = lifetimes.index.to_numpy()
    rng = np.random.default_rng(split_seed)
    n_strata = min(5, len(units))
    order = np.argsort(lifetimes.to_numpy(), kind="stable")
    strata = np.empty(len(units), dtype=np.int64)
    strata[order] = np.floor(
        np.arange(len(units)) * n_strata / len(units)
    ).astype(np.int64)
    selected = []
    for stratum in range(n_strata):
        members = units[strata == stratum]
        members = rng.permutation(members)
        count = max(1, int(round(len(members) * val_fraction)))
        selected.extend(members[:count].tolist())
    val_units = np.sort(np.asarray(selected))
    train_units = np.sort(np.setdiff1d(units, val_units))
    return train_units, val_units


def build_loaders(args, spec, train_df, test_df, truth_df, meta):
    train_units, val_units = split_units(
        train_df, args.split_seed, args.val_fraction
    )
    train_part = train_df[train_df["unit"].isin(train_units)].reset_index(drop=True)
    val_part = train_df[train_df["unit"].isin(val_units)].reset_index(drop=True)

    raw_cols = list(meta["raw_cols"])
    trend_cols = meta.get("trend_cols")
    online_trend = args.dataset in {"FD001", "FD003"}

    common = dict(
        raw_cols=raw_cols,
        trend_cols=trend_cols,
        seq_len=args.seq_len,
        online_trend=online_trend,
        trend_window=11,
    )
    train_ds = WindowDataset(train_part, augment=True, **common)
    val_ds = WindowDataset(val_part, augment=False, **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    test_loader = test_ds = None
    if not args.validation_only:
        external = truth_df["RUL"].to_numpy(dtype=np.float32)
        test_ds = WindowDataset(
            test_df, augment=False, external_rul=external, **common
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
    return train_loader, val_loader, test_loader, raw_cols, train_units, val_units


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def cap_exposure(target: np.ndarray, cap: float) -> float:
    """Model-free lower bound on hidden original-scale RMSE energy."""
    return float(np.sqrt(np.mean(np.maximum(target - cap, 0.0) ** 2)))


def nasa_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric score; late predictions are penalised more."""
    error = np.asarray(prediction) - np.asarray(target)
    terms = np.where(
        error < 0.0,
        np.exp(np.minimum(-error / 13.0, 50.0)) - 1.0,
        np.exp(np.minimum(error / 10.0, 50.0)) - 1.0,
    )
    return float(np.sum(terms))


def decision_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    thresholds=(20.0, 30.0, 50.0),
):
    """Directional error and threshold-based maintenance decision diagnostics."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    error = prediction - target
    metrics = {
        "mean_error": float(np.mean(error)),
        "late_prediction_fraction": float(np.mean(error > 0.0)),
        "early_prediction_fraction": float(np.mean(error < 0.0)),
    }
    for threshold in thresholds:
        due = target <= threshold
        predicted_due = prediction <= threshold
        delayed = due & ~predicted_due
        premature = ~due & predicted_due
        label = f"{int(threshold)}"
        metrics[f"delayed_action_rate_at_{label}"] = float(
            delayed.sum() / max(int(due.sum()), 1)
        )
        metrics[f"premature_action_rate_at_{label}"] = float(
            premature.sum() / max(int((~due).sum()), 1)
        )
    return metrics


def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    train_cap: float,
    reporting_cap: float,
    score_prediction_cap: str,
):
    model.eval()
    predictions, originals = [], []
    with torch.no_grad():
        for x, y_original, length in loader:
            x = x.to(DEVICE)
            length = length.to(DEVICE).unsqueeze(-1)
            with autocast():
                output = model(x, length)
                mu = output[0] if isinstance(output, tuple) else output
            predictions.append(mu.float().cpu().numpy())
            originals.append(y_original.numpy())
    support_cap = (
        reporting_cap if score_prediction_cap == "reference" else train_cap
    )
    raw_pred = np.nan_to_num(
        np.concatenate(predictions),
        nan=train_cap / 2.0,
        posinf=support_cap,
        neginf=0,
    )
    original = np.concatenate(originals)
    own_pred = np.clip(raw_pred, 0.0, train_cap)
    reference_pred = np.clip(raw_pred, 0.0, support_cap)
    own_target = np.clip(original, 0.0, train_cap)
    reference_target = np.clip(original, 0.0, reporting_cap)
    return {
        "own_rmse": rmse(own_pred, own_target),
        "reference_rmse": rmse(reference_pred, reference_target),
        "original_rmse": rmse(reference_pred, original),
        "saturation_fraction": float(np.mean(original > train_cap)),
        "exposure_bound": cap_exposure(original, train_cap),
    }


def build_model(args, num_sensors: int):
    return DualStreamTCN_v3(
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


def train_seed(args, raw_cols, train_loader, val_loader, checkpoint: Path, seed: int):
    set_seed(seed)
    model = build_model(args, len(raw_cols))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler()
    huber = nn.HuberLoss(delta=args.huber_delta)

    selected_key = args.checkpoint_metric
    best_value = float("inf")
    best_metrics = None
    best_epoch = 0
    stale = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y_original, length in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = torch.clamp(
                y_original.to(DEVICE, non_blocking=True), 0.0, args.train_cap
            )
            length = length.to(DEVICE, non_blocking=True).unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                mu, logvar = model(x, length)
                if not args.disable_het and logvar is not None:
                    loss = gaussian_nll_loss(mu, logvar, y)
                else:
                    loss = huber(mu, y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()

        metrics = validation_metrics(
            model,
            val_loader,
            args.train_cap,
            args.reporting_cap,
            args.score_prediction_cap,
        )
        selected_value = metrics[selected_key]
        if selected_value < best_value - 1e-8:
            best_value = selected_value
            best_metrics = metrics
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(
                f"  seed={seed} epoch={epoch:03d} loss={mean_loss:.4f} "
                f"own={metrics['own_rmse']:.3f} "
                f"ref={metrics['reference_rmse']:.3f} "
                f"best-{selected_key}={best_value:.3f}"
            )
        if stale >= args.patience:
            break

    if best_metrics is None or not checkpoint.exists():
        raise RuntimeError(f"Seed {seed} produced no valid checkpoint")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    # Recompute to guarantee metrics correspond to the saved state.
    best_metrics = validation_metrics(
        model,
        val_loader,
        args.train_cap,
        args.reporting_cap,
        args.score_prediction_cap,
    )
    return model, {"seed": seed, "best_epoch": best_epoch, **best_metrics}


def test_unit_predictions(model, loader, test_df, args):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for x, _, length in loader:
            x = x.to(DEVICE)
            length = length.to(DEVICE).unsqueeze(-1)
            with autocast():
                output = model(x, length)
                mu = output[0] if isinstance(output, tuple) else output
            all_predictions.append(mu.float().cpu().numpy())
    window_predictions = np.concatenate(all_predictions)

    unit_predictions = []
    cursor = 0
    for _, group in sorted(test_df.groupby("unit"), key=lambda pair: pair[0]):
        n_windows = max(1, len(group) - args.seq_len + 1)
        trajectory = window_predictions[cursor : cursor + n_windows]
        smoothed = pd.Series(trajectory).ewm(
            span=args.ema_span, adjust=False
        ).mean()
        unit_predictions.append(float(smoothed.iloc[-1]))
        cursor += n_windows
    return np.nan_to_num(
        unit_predictions,
        nan=args.train_cap / 2.0,
        posinf=max(args.train_cap, args.reporting_cap),
        neginf=0.0,
    )


def aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument("--output_root", type=Path, default=Path(r"E:\NASA_Lab\ress_strict"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--variant", default="V_A", choices=["V_A", "V_Aprime"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=20260724)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--train_cap", type=float, required=True)
    parser.add_argument("--reporting_cap", type=float, default=125.0)
    parser.add_argument(
        "--score_prediction_cap",
        choices=["train", "reference"],
        default="train",
        help=(
            "'train' reproduces joint target/output capping; 'reference' evaluates "
            "all candidates with the common output clip [0, reporting_cap]."
        ),
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=["own_rmse", "reference_rmse"],
        default="reference_rmse",
    )
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--ema_span", type=int, default=3)
    parser.add_argument("--disable_patch", action="store_true")
    parser.add_argument("--disable_xstream", action="store_true")
    parser.add_argument("--disable_pe", action="store_true")
    parser.add_argument("--disable_trans", action="store_true")
    parser.add_argument("--disable_skfusion", action="store_true")
    parser.add_argument("--disable_het", action="store_true")
    parser.add_argument("--disable_length", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = DEFAULTS[args.dataset]
    args.seq_len = args.seq_len or spec["seq_len"]
    args.lr = args.lr or spec["lr"]
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if not (0 < args.train_cap <= args.reporting_cap):
        raise ValueError("Require 0 < train_cap <= reporting_cap")

    print(
        f"dataset={args.dataset} device={DEVICE} train_cap={args.train_cap:g} "
        f"reporting_cap={args.reporting_cap:g} objective={args.checkpoint_metric} "
        f"score_prediction_cap={args.score_prediction_cap} "
        f"validation_only={args.validation_only}"
    )
    train_df, test_df, truth_df, meta, fingerprints = load_data(
        args.data_root, args.dataset, args.validation_only
    )
    (
        train_loader,
        val_loader,
        test_loader,
        raw_cols,
        train_units,
        val_units,
    ) = build_loaders(args, spec, train_df, test_df, truth_df, meta)

    checkpoints = args.output_root / "checkpoints" / args.dataset
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    test_predictions = []
    start = time.time()
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        checkpoint = checkpoints / f"{args.tag}_seed{seed}.pth"
        model, seed_result = train_seed(
            args, raw_cols, train_loader, val_loader, checkpoint, seed
        )
        per_seed.append(seed_result)
        if not args.validation_only:
            test_predictions.append(
                test_unit_predictions(model, test_loader, test_df, args)
            )

    result = {
        "protocol": "strict-rul-cap-v2",
        "dataset": args.dataset,
        "tag": args.tag,
        "device": DEVICE,
        "train_cap": args.train_cap,
        "reporting_cap": args.reporting_cap,
        "checkpoint_metric": args.checkpoint_metric,
        "score_prediction_cap": args.score_prediction_cap,
        "validation_only": args.validation_only,
        "split_seed": args.split_seed,
        "model_seeds": [item["seed"] for item in per_seed],
        "train_units": [int(value) for value in train_units],
        "validation_units": [int(value) for value in val_units],
        "config": {
            "variant": args.variant,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "huber_delta": args.huber_delta,
            "ema_span": args.ema_span,
            "disable_patch": args.disable_patch,
            "disable_xstream": args.disable_xstream,
            "disable_pe": args.disable_pe,
            "disable_trans": args.disable_trans,
            "disable_skfusion": args.disable_skfusion,
            "disable_het": args.disable_het,
            "disable_length": args.disable_length,
        },
        "data_fingerprints": fingerprints,
        "per_seed_validation": per_seed,
        "validation_summary": {
            key: aggregate([item[key] for item in per_seed])
            for key in [
                "own_rmse",
                "reference_rmse",
                "original_rmse",
                "saturation_fraction",
                "exposure_bound",
            ]
        },
        "elapsed_seconds": time.time() - start,
    }

    if not args.validation_only:
        raw_stack = np.asarray(test_predictions)
        original = truth_df["RUL"].to_numpy(dtype=np.float64)
        own_target = np.clip(original, 0.0, args.train_cap)
        reference_target = np.clip(original, 0.0, args.reporting_cap)
        support_cap = (
            args.reporting_cap
            if args.score_prediction_cap == "reference"
            else args.train_cap
        )
        own_stack = np.clip(raw_stack, 0.0, args.train_cap)
        reference_stack = np.clip(raw_stack, 0.0, support_cap)
        own_ensemble = np.median(own_stack, axis=0)
        reference_ensemble = np.median(reference_stack, axis=0)
        per_seed_test = []
        for seed, own_prediction, reference_prediction in zip(
            result["model_seeds"], own_stack, reference_stack
        ):
            per_seed_test.append(
                {
                    "seed": seed,
                    "own_rmse": rmse(own_prediction, own_target),
                    "reference_rmse": rmse(
                        reference_prediction, reference_target
                    ),
                    "original_rmse": rmse(reference_prediction, original),
                    "nasa_score_reference": nasa_score(
                        reference_prediction, reference_target
                    ),
                    "nasa_score_original": nasa_score(
                        reference_prediction, original
                    ),
                    **decision_metrics(reference_prediction, original),
                }
            )
        result["per_seed_test"] = per_seed_test
        result["test_summary"] = {
            key: aggregate([item[key] for item in per_seed_test])
            for key in [
                "own_rmse",
                "reference_rmse",
                "original_rmse",
                "nasa_score_reference",
                "nasa_score_original",
                "mean_error",
                "late_prediction_fraction",
                "early_prediction_fraction",
                "delayed_action_rate_at_20",
                "premature_action_rate_at_20",
                "delayed_action_rate_at_30",
                "premature_action_rate_at_30",
                "delayed_action_rate_at_50",
                "premature_action_rate_at_50",
            ]
        }
        result["ensemble_test"] = {
            "own_rmse": rmse(own_ensemble, own_target),
            "reference_rmse": rmse(reference_ensemble, reference_target),
            "original_rmse": rmse(reference_ensemble, original),
            "nasa_score_reference": nasa_score(
                reference_ensemble, reference_target
            ),
            "nasa_score_original": nasa_score(reference_ensemble, original),
            **decision_metrics(reference_ensemble, original),
            "saturation_fraction": float(np.mean(original > args.train_cap)),
            "exposure_bound": cap_exposure(original, args.train_cap),
            "predictions": [float(value) for value in reference_ensemble],
            "own_support_predictions": [
                float(value) for value in own_ensemble
            ],
            "original_targets": [float(value) for value in original],
        }

    result_path = results_dir / f"{args.dataset}_{args.tag}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
