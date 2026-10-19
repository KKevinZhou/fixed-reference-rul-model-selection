"""Confirmatory, split-safe C-MAPSS experiment for RUL-cap selection.

The implementation enforces the protocol used by the revised manuscript:

* raw C-MAPSS files are split by engine before fitting any preprocessor;
* validation engines are converted to deterministic truncated trajectories;
* every validation engine contributes one endpoint prediction, as in the
  official C-MAPSS test task;
* the training-label cap, candidate output support, and fixed reporting
  support are distinct;
* one training run saves separate checkpoints for the self-scaled and
  fixed-reference objectives;
* validation-only runs never open the official test or truth files.

This file intentionally does not reuse the legacy processed pickles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path

# Required by CUDA for reproducible matrix multiplications.  It must be set
# before torch initialises cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import bootstrap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baselines.models_v3 import DualStreamTCN_v3


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLUMNS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s_{index}" for index in range(1, 22)]
)
SENSORS = [
    "s_2",
    "s_3",
    "s_4",
    "s_6",
    "s_7",
    "s_8",
    "s_9",
    "s_11",
    "s_12",
    "s_13",
    "s_14",
    "s_15",
    "s_17",
    "s_20",
    "s_21",
]
SETTINGS = ["setting_1", "setting_2", "setting_3"]
DEFAULTS = {
    "FD001": {"seq_len": 50, "lr": 5e-4, "multi_condition": False},
    "FD002": {"seq_len": 50, "lr": 1e-3, "multi_condition": True},
    "FD003": {"seq_len": 80, "lr": 1e-3, "multi_condition": False},
    "FD004": {"seq_len": 50, "lr": 5e-4, "multi_condition": True},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, state_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(state_dict, temporary)
    os.replace(temporary, path)


def raw_paths(data_root: Path, dataset: str) -> dict[str, Path]:
    base = data_root / "Experiment_FD001" / "CMAPSSData"
    return {
        "train": base / f"train_{dataset}.txt",
        "test": base / f"test_{dataset}.txt",
        "truth": base / f"RUL_{dataset}.txt",
    }


def read_raw_training(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    maximum = frame.groupby("unit", sort=False)["cycle"].transform("max")
    frame["RUL_ORIGINAL"] = (maximum - frame["cycle"]).astype(np.float32)
    return frame


def read_raw_test(test_path: Path, truth_path: Path):
    frame = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    truth = pd.read_csv(truth_path, sep=r"\s+", header=None, names=["RUL"])
    units = np.sort(frame["unit"].unique()).astype(int)
    expected = np.arange(1, len(truth) + 1, dtype=int)
    if not np.array_equal(units, expected):
        raise ValueError("Test units are not the expected contiguous identifiers")
    truth["unit"] = expected
    target_map = {
        int(row.unit): float(row.RUL)
        for row in truth[["unit", "RUL"]].itertuples(index=False)
    }
    return frame, truth, target_map


def split_units(frame: pd.DataFrame, split_seed: int, val_fraction: float):
    """Lifetime-stratified deterministic group split."""
    lifetimes = frame.groupby("unit", sort=True)["cycle"].max().sort_index()
    units = lifetimes.index.to_numpy(dtype=int)
    rng = np.random.default_rng(split_seed)
    n_strata = min(5, len(units))
    order = np.argsort(lifetimes.to_numpy(), kind="stable")
    strata = np.empty(len(units), dtype=np.int64)
    strata[order] = np.floor(
        np.arange(len(units)) * n_strata / len(units)
    ).astype(np.int64)
    selected: list[int] = []
    for stratum in range(n_strata):
        members = rng.permutation(units[strata == stratum])
        count = max(1, int(round(len(members) * val_fraction)))
        selected.extend(int(value) for value in members[:count])
    validation = np.sort(np.asarray(selected, dtype=int))
    training = np.sort(np.setdiff1d(units, validation))
    if set(training).intersection(validation):
        raise AssertionError("Training and validation units overlap")
    return training, validation


def validation_endpoints(
    frame: pd.DataFrame,
    validation_units: np.ndarray,
    seq_len: int,
    reporting_cap: float,
    endpoint_seed: int,
):
    """Create one deterministic, range-stratified endpoint per held-out unit."""
    rng = np.random.default_rng(endpoint_seed)
    quantiles = (np.arange(len(validation_units)) + 0.5) / len(validation_units)
    quantiles = rng.permutation(quantiles)
    targets: dict[int, float] = {}
    cutoffs: dict[int, int] = {}
    schedule = []
    for unit, quantile in zip(validation_units, quantiles):
        lifetime = int(frame.loc[frame["unit"] == unit, "cycle"].max())
        minimum_observed = min(lifetime, max(seq_len, 30))
        maximum_rul = max(0.0, min(reporting_cap, lifetime - minimum_observed))
        minimum_rul = min(5.0, maximum_rul)
        target = float(
            round(minimum_rul + quantile * (maximum_rul - minimum_rul))
        )
        cutoff = int(lifetime - target)
        if cutoff <= 0:
            raise AssertionError("Invalid validation cutoff")
        targets[int(unit)] = target
        cutoffs[int(unit)] = cutoff
        schedule.append(
            {
                "unit": int(unit),
                "lifetime": lifetime,
                "observed_cycles": cutoff,
                "original_rul": target,
                "quantile": float(quantile),
            }
        )
    prefix = frame[
        frame.apply(
            lambda row: (
                int(row["unit"]) in cutoffs
                and int(row["cycle"]) <= cutoffs[int(row["unit"])]
            ),
            axis=1,
        )
    ].copy()
    observed = set(prefix["unit"].astype(int).unique())
    if observed != set(int(value) for value in validation_units):
        raise AssertionError("A validation unit disappeared during truncation")
    return prefix.reset_index(drop=True), targets, schedule


class FoldPreprocessor:
    """Preprocessor fitted only on the training units of one split."""

    def __init__(self, multi_condition: bool, random_state: int):
        self.multi_condition = multi_condition
        self.random_state = random_state
        self.kmeans = None
        self.global_scaler = None
        self.cluster_scalers: dict[int, StandardScaler] = {}
        self.trend_cols = [f"{sensor}_trend" for sensor in SENSORS]

    def fit(self, frame: pd.DataFrame):
        if self.multi_condition:
            self.kmeans = KMeans(
                n_clusters=6,
                random_state=self.random_state,
                n_init=20,
            )
            labels = self.kmeans.fit_predict(frame[SETTINGS])
            for cluster in range(6):
                subset = frame.loc[labels == cluster, SENSORS]
                if subset.empty:
                    raise RuntimeError(f"Empty operating cluster {cluster}")
                self.cluster_scalers[cluster] = StandardScaler().fit(subset)
        else:
            self.global_scaler = StandardScaler().fit(frame[SENSORS])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        if self.multi_condition:
            labels = self.kmeans.predict(transformed[SETTINGS])
            for cluster in range(6):
                mask = labels == cluster
                if np.any(mask):
                    transformed.loc[mask, SENSORS] = self.cluster_scalers[
                        cluster
                    ].transform(transformed.loc[mask, SENSORS])
        else:
            transformed[SENSORS] = self.global_scaler.transform(
                transformed[SENSORS]
            )

        for unit, indices in transformed.groupby("unit", sort=True).groups.items():
            raw = transformed.loc[indices, SENSORS]
            causal_trend = raw.rolling(window=11, min_periods=1).mean()
            transformed.loc[indices, self.trend_cols] = causal_trend.to_numpy()
        if not np.isfinite(
            transformed[SENSORS + self.trend_cols].to_numpy()
        ).all():
            raise FloatingPointError("Non-finite value after preprocessing")
        return transformed

    def fingerprint(self) -> str:
        return sha256_bytes(pickle.dumps(self, protocol=5))


class WindowDataset(Dataset):
    """Sliding windows with explicit unit identifiers."""

    def __init__(
        self,
        frame: pd.DataFrame,
        seq_len: int,
        endpoint_targets: dict[int, float] | None,
        augment: bool,
        stride: int = 1,
    ):
        self.augment = augment
        features = SENSORS + [f"{sensor}_trend" for sensor in SENSORS]
        xs, ys, lengths, units = [], [], [], []
        for unit in sorted(frame["unit"].unique()):
            group = frame[frame["unit"] == unit].sort_values("cycle")
            values = group[features].to_numpy(dtype=np.float32)
            if endpoint_targets is None:
                targets = group["RUL_ORIGINAL"].to_numpy(dtype=np.float32)
            else:
                targets = np.full(
                    len(group), endpoint_targets[int(unit)], dtype=np.float32
                )
            n_windows = max(1, len(values) - seq_len + 1)
            starts = list(range(0, n_windows, stride))
            if starts[-1] != n_windows - 1:
                starts.append(n_windows - 1)
            for start in starts:
                end = start + seq_len
                if end > len(values):
                    pad = end - len(values)
                    sequence = np.concatenate(
                        [np.repeat(values[:1], pad, axis=0), values[start:]],
                        axis=0,
                    )
                    effective_length = len(values) - start
                    target = targets[-1]
                else:
                    sequence = values[start:end]
                    effective_length = seq_len
                    target = targets[end - 1]
                xs.append(sequence)
                ys.append(target)
                lengths.append(
                    math.log(max(effective_length, 1)) / math.log(seq_len)
                )
                units.append(int(unit))
        self.x = np.asarray(xs, dtype=np.float32)
        self.y = np.asarray(ys, dtype=np.float32)
        self.length = np.asarray(lengths, dtype=np.float32)
        self.unit = np.asarray(units, dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        value = self.x[index]
        if self.augment:
            value = value.copy()
            half = value.shape[1] // 2
            if np.random.rand() > 0.5:
                value[:, :half] += np.random.normal(
                    0.0, 0.02, value[:, :half].shape
                ).astype(np.float32)
            if np.random.rand() > 0.5:
                value[:, :half] *= np.random.uniform(0.95, 1.05)
        return (
            torch.from_numpy(value),
            torch.tensor(self.y[index], dtype=torch.float32),
            torch.tensor(self.length[index], dtype=torch.float32),
            torch.tensor(self.unit[index], dtype=torch.int64),
        )


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.GroupNorm(4, channels),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.GroupNorm(4, channels),
        )

    def forward(self, value):
        return value + self.network(value)


class CompactTCN(nn.Module):
    """Small transparent carrier used for the confirmatory cap study."""

    def __init__(self, input_channels: int, channels: int = 32):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, value, length=None):
        hidden = self.input_projection(value.transpose(1, 2))
        hidden = self.blocks(hidden)
        prediction = self.head(hidden[:, :, -1]).squeeze(-1)
        return prediction, None


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=True,
        num_workers=0,
        generator=generator,
    )


def build_model(args):
    if args.carrier == "compact_tcn":
        return CompactTCN(input_channels=2 * len(SENSORS)).to(DEVICE)
    return DualStreamTCN_v3(
        num_sensors=len(SENSORS),
        d_model=128,
        variant="V_A",
        disable_patch=False,
        disable_xstream=False,
        disable_pe=False,
        disable_trans=True,
        disable_skfusion=True,
        disable_het=True,
        disable_length=True,
    ).to(DEVICE)


def rmse(prediction, target) -> float:
    return float(
        np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))
    )


def nasa_score(prediction, target) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    terms = np.where(
        error < 0,
        np.exp(np.minimum(-error / 13.0, 50.0)) - 1.0,
        np.exp(np.minimum(error / 10.0, 50.0)) - 1.0,
    )
    return float(np.sum(terms))


def exposure(target, cap) -> float:
    return float(
        np.sqrt(np.mean(np.maximum(np.asarray(target) - cap, 0.0) ** 2))
    )


def endpoint_predictions(model, loader, ema_span: int):
    model.eval()
    predictions, units = [], []
    with torch.no_grad():
        for x, _, length, unit in loader:
            x = x.to(DEVICE, non_blocking=True)
            length = length.to(DEVICE, non_blocking=True).unsqueeze(-1)
            with autocast():
                output = model(x, length)
                mu = output[0] if isinstance(output, tuple) else output
            values = mu.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite endpoint prediction")
            predictions.append(values)
            units.append(unit.numpy())
    predictions = np.concatenate(predictions)
    units = np.concatenate(units)
    endpoint = {}
    for unit in np.unique(units):
        trajectory = predictions[units == unit]
        endpoint[int(unit)] = float(
            pd.Series(trajectory)
            .ewm(span=ema_span, adjust=False)
            .mean()
            .iloc[-1]
        )
    return endpoint


def decision_metrics(prediction, target):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    error = prediction - target
    result = {
        "mean_error": float(np.mean(error)),
        "late_prediction_fraction": float(np.mean(error > 0)),
        "early_prediction_fraction": float(np.mean(error < 0)),
    }
    for threshold in (20.0, 30.0, 50.0):
        due = target <= threshold
        predicted_due = prediction <= threshold
        label = str(int(threshold))
        result[f"delayed_action_rate_at_{label}"] = float(
            np.sum(due & ~predicted_due) / max(int(np.sum(due)), 1)
        )
        result[f"premature_action_rate_at_{label}"] = float(
            np.sum(~due & predicted_due) / max(int(np.sum(~due)), 1)
        )
    return result


def metric_bundle(
    raw_endpoint: dict[int, float],
    target_map: dict[int, float],
    train_cap: float,
    reporting_cap: float,
):
    units = np.asarray(sorted(target_map), dtype=int)
    raw_prediction = np.asarray(
        [raw_endpoint[int(unit)] for unit in units], dtype=np.float64
    )
    original = np.asarray(
        [target_map[int(unit)] for unit in units], dtype=np.float64
    )
    candidate_prediction = np.clip(raw_prediction, 0.0, train_cap)
    common_prediction = np.clip(raw_prediction, 0.0, reporting_cap)
    own_target = np.clip(original, 0.0, train_cap)
    reference_target = np.clip(original, 0.0, reporting_cap)
    return {
        "own_rmse": rmse(candidate_prediction, own_target),
        "reference_rmse": rmse(common_prediction, reference_target),
        "original_rmse": rmse(common_prediction, original),
        "nasa_score_reference": nasa_score(
            common_prediction, reference_target
        ),
        "nasa_score_original": nasa_score(common_prediction, original),
        "saturation_fraction": float(np.mean(original > train_cap)),
        "exposure_bound": exposure(original, train_cap),
        "joint_support_penalty": exposure(reference_target, train_cap),
        **decision_metrics(common_prediction, original),
        "units": [int(value) for value in units],
        "raw_predictions": [float(value) for value in raw_prediction],
        "candidate_support_predictions": [
            float(value) for value in candidate_prediction
        ],
        "common_support_predictions": [
            float(value) for value in common_prediction
        ],
        "original_targets": [float(value) for value in original],
    }


def scalar_metrics(bundle: dict) -> dict:
    return {
        key: value
        for key, value in bundle.items()
        if isinstance(value, (int, float))
    }


def train_seed(
    args,
    seed: int,
    train_dataset: WindowDataset,
    endpoint_loader,
    endpoint_targets: dict[int, float],
    checkpoint_dir: Path,
):
    set_seed(seed)
    train_loader = make_loader(
        train_dataset, args.batch, shuffle=True, seed=seed
    )
    model = build_model(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler()
    loss_function = nn.HuberLoss(delta=args.huber_delta)
    paths = {
        "self_scaled": checkpoint_dir
        / f"{args.tag}_seed{seed}_self_scaled.pth",
        "fixed_reference": checkpoint_dir
        / f"{args.tag}_seed{seed}_fixed_reference.pth",
    }
    selected_keys = {
        "self_scaled": "own_rmse",
        "fixed_reference": "reference_rmse",
    }
    best = {name: float("inf") for name in paths}
    best_epoch = {name: 0 for name in paths}
    stale = {name: 0 for name in paths}
    nonfinite_batches = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, original_target, length, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            target = torch.clamp(
                original_target.to(DEVICE, non_blocking=True),
                0.0,
                args.train_cap,
            )
            length = length.to(DEVICE, non_blocking=True).unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(x, length)
                prediction = output[0] if isinstance(output, tuple) else output
                loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()

        should_validate = (
            epoch == 1
            or epoch % args.validation_interval == 0
            or epoch == args.epochs
        )
        if should_validate:
            endpoint = endpoint_predictions(
                model, endpoint_loader, args.ema_span
            )
            metrics = metric_bundle(
                endpoint,
                endpoint_targets,
                args.train_cap,
                args.reporting_cap,
            )
            for name, key in selected_keys.items():
                value = metrics[key]
                if value < best[name] - 1e-8:
                    best[name] = value
                    best_epoch[name] = epoch
                    stale[name] = 0
                    atomic_checkpoint(paths[name], model.state_dict())
                else:
                    stale[name] += 1

        if should_validate and (epoch == 1 or epoch % 10 == 0):
            print(
                f"seed={seed} epoch={epoch:03d} "
                f"loss={np.mean(losses):.4f} "
                f"own={metrics['own_rmse']:.3f} "
                f"ref={metrics['reference_rmse']:.3f}",
                flush=True,
            )
        if should_validate and all(
            value >= args.patience for value in stale.values()
        ):
            break

    output = {
        "seed": seed,
        "nonfinite_batches": nonfinite_batches,
        "checkpoints": {},
    }
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing checkpoint {path}")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        endpoint = endpoint_predictions(model, endpoint_loader, args.ema_span)
        bundle = metric_bundle(
            endpoint,
            endpoint_targets,
            args.train_cap,
            args.reporting_cap,
        )
        output["checkpoints"][name] = {
            "best_epoch": best_epoch[name],
            "checkpoint_sha256": sha256_file(path),
            "metrics": bundle,
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, paths


def aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def summarise_validation(per_seed: list[dict]):
    summary = {}
    for objective in ("self_scaled", "fixed_reference"):
        metric_names = scalar_metrics(
            per_seed[0]["checkpoints"][objective]["metrics"]
        )
        summary[objective] = {
            key: aggregate(
                [
                    item["checkpoints"][objective]["metrics"][key]
                    for item in per_seed
                ]
            )
            for key in metric_names
        }
        summary[objective]["best_epoch"] = aggregate(
            [
                item["checkpoints"][objective]["best_epoch"]
                for item in per_seed
            ]
        )
    return summary


def bootstrap_rmse_ci(prediction, target, seed: int):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    squared = (prediction - target) ** 2
    if len(squared) < 2:
        return [float(np.sqrt(np.mean(squared)))] * 2
    result = bootstrap(
        (squared,),
        statistic=lambda values: np.sqrt(np.mean(values)),
        n_resamples=2000,
        confidence_level=0.95,
        random_state=seed,
        method="percentile",
    )
    return [
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    ]


def evaluate_test(
    args,
    per_seed,
    checkpoint_paths,
    test_dataset,
    target_map,
):
    test_loader = make_loader(
        test_dataset, args.batch * 2, shuffle=False, seed=0
    )
    output = {}
    for objective in ("self_scaled", "fixed_reference"):
        seed_bundles = []
        common_predictions = []
        candidate_predictions = []
        for seed_index, seed_result in enumerate(per_seed):
            seed = seed_result["seed"]
            model = build_model(args)
            model.load_state_dict(
                torch.load(
                    checkpoint_paths[seed_index][objective],
                    map_location=DEVICE,
                )
            )
            endpoint = endpoint_predictions(model, test_loader, args.ema_span)
            bundle = metric_bundle(
                endpoint,
                target_map,
                args.train_cap,
                args.reporting_cap,
            )
            seed_bundles.append(bundle)
            common_predictions.append(bundle["common_support_predictions"])
            candidate_predictions.append(
                bundle["candidate_support_predictions"]
            )
            del model
        units = seed_bundles[0]["units"]
        original = np.asarray(seed_bundles[0]["original_targets"])
        common_ensemble = np.median(
            np.asarray(common_predictions), axis=0
        )
        candidate_ensemble = np.median(
            np.asarray(candidate_predictions), axis=0
        )
        raw_endpoint = {
            int(unit): float(value)
            for unit, value in zip(units, common_ensemble)
        }
        ensemble = metric_bundle(
            raw_endpoint,
            target_map,
            args.train_cap,
            args.reporting_cap,
        )
        ensemble["common_support_predictions"] = [
            float(value) for value in common_ensemble
        ]
        ensemble["candidate_support_predictions"] = [
            float(value) for value in candidate_ensemble
        ]
        ensemble["reference_rmse_bootstrap_95ci"] = bootstrap_rmse_ci(
            common_ensemble,
            np.clip(original, 0.0, args.reporting_cap),
            args.bootstrap_seed,
        )
        ensemble["original_rmse_bootstrap_95ci"] = bootstrap_rmse_ci(
            common_ensemble, original, args.bootstrap_seed
        )
        scalar_names = scalar_metrics(seed_bundles[0])
        output[objective] = {
            "per_seed": [
                {
                    "seed": per_seed[index]["seed"],
                    **scalar_metrics(bundle),
                }
                for index, bundle in enumerate(seed_bundles)
            ],
            "summary": {
                key: aggregate([bundle[key] for bundle in seed_bundles])
                for key in scalar_names
            },
            "ensemble": ensemble,
        }
    return output


def environment_manifest():
    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "device": DEVICE,
        "device_name": gpu,
        "cudnn": (
            str(torch.backends.cudnn.version())
            if torch.cuda.is_available()
            else None
        ),
        "deterministic_algorithms": True,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--campaign_id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train_cap", type=float, required=True)
    parser.add_argument("--reporting_cap", type=float, default=125.0)
    parser.add_argument("--split_seed", type=int, default=20260724)
    parser.add_argument("--endpoint_seed", type=int)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--ema_span", type=int, default=3)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--validation_interval", type=int, default=2)
    parser.add_argument("--bootstrap_seed", type=int, default=20260724)
    parser.add_argument(
        "--carrier",
        choices=["compact_tcn", "dualstream"],
        default="compact_tcn",
    )
    parser.add_argument("--validation_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = DEFAULTS[args.dataset]
    args.seq_len = args.seq_len or spec["seq_len"]
    args.lr = args.lr or spec["lr"]
    args.endpoint_seed = args.endpoint_seed or args.split_seed + 1000003
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if not (0 < args.train_cap <= args.reporting_cap):
        raise ValueError("Require 0 < train_cap <= reporting_cap")
    result_path = (
        args.output_root / "results" / f"{args.dataset}_{args.tag}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"Immutable result already exists: {result_path}")

    paths = raw_paths(args.data_root, args.dataset)
    train_raw = read_raw_training(paths["train"])
    training_units, validation_units = split_units(
        train_raw, args.split_seed, args.val_fraction
    )
    train_part_raw = train_raw[
        train_raw["unit"].isin(training_units)
    ].reset_index(drop=True)
    validation_prefix_raw, endpoint_targets, endpoint_schedule = (
        validation_endpoints(
            train_raw,
            validation_units,
            args.seq_len,
            args.reporting_cap,
            args.endpoint_seed,
        )
    )
    preprocessor = FoldPreprocessor(
        multi_condition=spec["multi_condition"],
        random_state=args.split_seed,
    ).fit(train_part_raw)
    train_part = preprocessor.transform(train_part_raw)
    validation_prefix = preprocessor.transform(validation_prefix_raw)
    train_dataset = WindowDataset(
        train_part,
        args.seq_len,
        endpoint_targets=None,
        augment=True,
        stride=args.train_stride,
    )
    validation_dataset = WindowDataset(
        validation_prefix,
        args.seq_len,
        endpoint_targets=endpoint_targets,
        augment=False,
        stride=1,
    )
    validation_loader = make_loader(
        validation_dataset, args.batch * 2, shuffle=False, seed=0
    )

    checkpoint_dir = (
        args.output_root / "checkpoints" / args.campaign_id / args.dataset
    )
    start = time.time()
    per_seed = []
    checkpoint_paths = []
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        seed_result, paths_by_objective = train_seed(
            args,
            seed,
            train_dataset,
            validation_loader,
            endpoint_targets,
            checkpoint_dir,
        )
        per_seed.append(seed_result)
        checkpoint_paths.append(paths_by_objective)

    data_fingerprints = {"train_sha256": sha256_file(paths["train"])}
    result = {
        "protocol": "confirmatory-rul-cap-v2",
        "campaign_id": args.campaign_id,
        "dataset": args.dataset,
        "tag": args.tag,
        "validation_only": args.validation_only,
        "train_cap": args.train_cap,
        "reporting_cap": args.reporting_cap,
        "prediction_supports": {
            "self_scaled": "clip(raw_prediction, 0, train_cap)",
            "fixed_reference": "clip(raw_prediction, 0, reporting_cap)",
        },
        "split_seed": args.split_seed,
        "endpoint_seed": args.endpoint_seed,
        "training_units": [int(value) for value in training_units],
        "validation_units": [int(value) for value in validation_units],
        "validation_endpoints": endpoint_schedule,
        "model_seeds": [item["seed"] for item in per_seed],
        "config": {
            "seq_len": args.seq_len,
            "lr": args.lr,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "huber_delta": args.huber_delta,
            "ema_span": args.ema_span,
            "train_stride": args.train_stride,
            "validation_interval": args.validation_interval,
            "val_fraction": args.val_fraction,
            "architecture": (
                "CompactTCN-32"
                if args.carrier == "compact_tcn"
                else (
                    "DualStreamTCN_v3; patch+xstream; "
                    "transformer/skfusion/heteroscedastic/length disabled"
                )
            ),
            "carrier": args.carrier,
            "preprocessing": (
                "training-fold-only condition normalisation and "
                "causal rolling-mean trend"
            ),
        },
        "data_fingerprints": data_fingerprints,
        "source_fingerprints": {
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
            "model_sha256": sha256_file(
                SCRIPT_DIR / "baselines" / "models_v3.py"
            ),
            "preprocessor_sha256": preprocessor.fingerprint(),
        },
        "environment": environment_manifest(),
        "per_seed_validation": per_seed,
        "validation_summary": summarise_validation(per_seed),
        "test_evaluated": False,
        "elapsed_seconds": time.time() - start,
    }

    if not args.validation_only:
        test_raw, truth, test_targets = read_raw_test(
            paths["test"], paths["truth"]
        )
        test_part = preprocessor.transform(test_raw)
        test_dataset = WindowDataset(
            test_part,
            args.seq_len,
            endpoint_targets=test_targets,
            augment=False,
            stride=1,
        )
        result["data_fingerprints"].update(
            test_sha256=sha256_file(paths["test"]),
            truth_sha256=sha256_file(paths["truth"]),
        )
        result["test_results"] = evaluate_test(
            args,
            per_seed,
            checkpoint_paths,
            test_dataset,
            test_targets,
        )
        result["test_evaluated"] = True
        result["test_unit_count"] = len(truth)

    result["elapsed_seconds"] = time.time() - start
    atomic_json(result_path, result)
    print(f"WROTE {result_path}", flush=True)


if __name__ == "__main__":
    main()
