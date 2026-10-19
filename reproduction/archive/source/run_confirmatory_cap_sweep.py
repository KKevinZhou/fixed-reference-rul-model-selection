"""Run and lock the confirmatory multi-split cap-selection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "confirmatory_rul_train.py"
MODEL = SCRIPT_DIR / "baselines" / "models_v3.py"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
CAPS = [90, 100, 110, 115, 125]
SPLIT_SEEDS = [20260724, 20260725, 20260726]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--validation_interval", type=int, default=2)
    parser.add_argument("--carrier", default="compact_tcn")
    return parser.parse_args()


def build_manifest(args):
    train_hashes = {}
    base = args.data_root / "Experiment_FD001" / "CMAPSSData"
    for dataset in DATASETS:
        path = base / f"train_{dataset}.txt"
        train_hashes[dataset] = sha256_file(path)
    definition = {
        "protocol": "confirmatory-cap-sweep-v2",
        "datasets": DATASETS,
        "caps": CAPS,
        "reporting_cap": 125,
        "split_seeds": SPLIT_SEEDS,
        "endpoint_seed_rule": "split_seed + 1000003",
        "model_seeds": list(
            range(args.seed_base, args.seed_base + args.seeds)
        ),
        "epochs": args.epochs,
        "patience": args.patience,
        "train_stride": args.train_stride,
        "validation_interval": args.validation_interval,
        "carrier": args.carrier,
        "trainer_sha256": sha256_file(TRAINER),
        "model_sha256": sha256_file(MODEL),
        "raw_train_sha256": train_hashes,
        "selection_rule": (
            "minimum split-mean objective; exact ties prefer the larger cap"
        ),
        "test_evaluated": False,
    }
    definition["campaign_id"] = "ccv2-" + stable_hash(definition)[:16]
    return definition


def validate_cached(path: Path, manifest: dict, dataset, cap, split_seed):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        result.get("protocol") == "confirmatory-rul-cap-v2"
        and result.get("campaign_id") == manifest["campaign_id"]
        and result.get("dataset") == dataset
        and float(result.get("train_cap")) == float(cap)
        and int(result.get("split_seed")) == int(split_seed)
        and result.get("validation_only") is True
        and result.get("model_seeds") == manifest["model_seeds"]
        and result.get("config", {}).get("epochs") == manifest["epochs"]
        and result.get("config", {}).get("patience") == manifest["patience"]
        and result.get("config", {}).get("train_stride")
        == manifest["train_stride"]
        and result.get("config", {}).get("validation_interval")
        == manifest["validation_interval"]
        and result.get("config", {}).get("carrier") == manifest["carrier"]
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == manifest["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == manifest["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == manifest["raw_train_sha256"][dataset]
        and result.get("test_evaluated") is False
    )


def run_one(args, manifest, dataset, cap, split_seed):
    tag = f"cap{cap}_split{split_seed}"
    result_path = (
        args.output_root / "results" / f"{dataset}_{tag}.json"
    )
    if validate_cached(
        result_path, manifest, dataset, cap, split_seed
    ):
        print(f"SKIP {dataset} cap={cap} split={split_seed}", flush=True)
        return result_path
    if result_path.exists():
        raise RuntimeError(
            f"Cached result does not match campaign manifest: {result_path}"
        )
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--dataset",
        dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--campaign_id",
        manifest["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(cap),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(split_seed),
        "--endpoint_seed",
        str(split_seed + 1000003),
        "--seeds",
        str(args.seeds),
        "--seed_base",
        str(args.seed_base),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--train_stride",
        str(args.train_stride),
        "--validation_interval",
        str(args.validation_interval),
        "--carrier",
        args.carrier,
        "--validation_only",
    ]
    log_path = (
        args.output_root
        / "logs"
        / f"{dataset}_{tag}_{manifest['campaign_id']}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        attempt = 2
        while True:
            candidate = log_path.with_name(
                f"{log_path.stem}_attempt{attempt}{log_path.suffix}"
            )
            if not candidate.exists():
                log_path = candidate
                break
            attempt += 1
    print(f"RUN {dataset} cap={cap} split={split_seed}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Run failed; see {log_path}")
    if not validate_cached(
        result_path, manifest, dataset, cap, split_seed
    ):
        raise RuntimeError(f"Result failed manifest validation: {result_path}")
    return result_path


def lock_decisions(args, manifest, result_paths):
    rows = {}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (
            result["dataset"],
            int(result["train_cap"]),
            int(result["split_seed"]),
        )
        rows[key] = result
    expected = len(DATASETS) * len(CAPS) * len(SPLIT_SEEDS)
    if len(rows) != expected:
        raise RuntimeError(
            f"Incomplete candidate matrix: {len(rows)} of {expected}"
        )

    decisions = {}
    metric_definitions = {
        "self_scaled": "own_rmse",
        "fixed_reference": "reference_rmse",
    }
    for dataset in DATASETS:
        decisions[dataset] = {}
        for objective, metric in metric_definitions.items():
            candidates = []
            split_choices = []
            for split_seed in SPLIT_SEEDS:
                split_values = []
                for cap in CAPS:
                    value = rows[
                        (dataset, cap, split_seed)
                    ]["validation_summary"][objective][metric]["mean"]
                    split_values.append((float(value), cap))
                split_value, split_cap = min(
                    split_values, key=lambda item: (item[0], -item[1])
                )
                split_choices.append(
                    {
                        "split_seed": split_seed,
                        "cap": split_cap,
                        "objective_value": split_value,
                    }
                )
            for cap in CAPS:
                values = [
                    float(
                        rows[(dataset, cap, split_seed)][
                            "validation_summary"
                        ][objective][metric]["mean"]
                    )
                    for split_seed in SPLIT_SEEDS
                ]
                candidates.append(
                    {
                        "cap": cap,
                        "split_values": values,
                        "mean": float(sum(values) / len(values)),
                        "std": float(
                            np_std(values)
                        ),
                    }
                )
            selected = min(
                candidates, key=lambda item: (item["mean"], -item["cap"])
            )
            decisions[dataset][objective] = {
                "metric": metric,
                "selected_cap": selected["cap"],
                "selection_value": selected["mean"],
                "split_choices": split_choices,
                "candidate_table": candidates,
            }
    lock = {
        "protocol": "confirmatory-cap-lock-v2",
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": stable_hash(manifest),
        "created_unix": time.time(),
        "selection_rule": manifest["selection_rule"],
        "decisions": decisions,
        "test_evaluated": False,
    }
    lock_path = args.output_root / "locked_cap_decisions.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            existing.get("campaign_id") != manifest["campaign_id"]
            or existing.get("decisions") != decisions
        ):
            raise RuntimeError("Existing immutable lock differs")
    else:
        exclusive_json(lock_path, lock)
    return lock_path, decisions


def np_std(values):
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = args.output_root / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "Output root belongs to a different immutable campaign"
            )
    else:
        exclusive_json(manifest_path, manifest)

    result_paths = []
    for dataset in DATASETS:
        for split_seed in SPLIT_SEEDS:
            for cap in CAPS:
                result_paths.append(
                    run_one(
                        args, manifest, dataset, cap, split_seed
                    )
                )
    lock_path, decisions = lock_decisions(
        args, manifest, result_paths
    )
    summary = {
        "status": "validation_complete",
        "campaign_id": manifest["campaign_id"],
        "completed_runs": len(result_paths),
        "lock_path": str(lock_path),
        "decisions": decisions,
        "test_evaluated": False,
        "completed_unix": time.time(),
    }
    atomic_json(args.output_root / "cap_sweep_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
