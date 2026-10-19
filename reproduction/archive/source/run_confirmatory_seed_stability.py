"""Add model-seed stability before creating the final cap-decision lock.

The three-split screening campaign uses model seed 42.  This stage reuses that
result and adds seeds 43 and 44 for every cap on the preregistered primary
split.  Final cap decisions use an equal-weight mean over the three grouped
splits: the primary-split value is first averaged over model seeds 42--44,
whereas each secondary split contributes its preregistered seed-42 value.
This prevents the additional seed runs from silently discarding split
sensitivity.
"""

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
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
CAPS = [90, 100, 110, 115, 125]
PRIMARY_SPLIT = 20260724


def stable_hash(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    return parser.parse_args()


def stability_manifest(screening_manifest: dict):
    definition = {
        "protocol": "confirmatory-model-seed-stability-v2",
        "screening_campaign_id": screening_manifest["campaign_id"],
        "screening_manifest_sha256": stable_hash(screening_manifest),
        "datasets": DATASETS,
        "caps": CAPS,
        "primary_split_seed": PRIMARY_SPLIT,
        "reused_model_seed": 42,
        "additional_model_seeds": [43, 44],
        "stability_script_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": screening_manifest["trainer_sha256"],
        "model_sha256": screening_manifest["model_sha256"],
        "epochs": screening_manifest["epochs"],
        "patience": screening_manifest["patience"],
        "train_stride": screening_manifest["train_stride"],
        "validation_interval": screening_manifest["validation_interval"],
        "carrier": screening_manifest["carrier"],
        "selection_rule": (
            "minimum equal-weight mean over three grouped splits; primary "
            "split value is the mean over model seeds 42--44 and each "
            "secondary split uses preregistered model seed 42; exact ties "
            "prefer larger cap"
        ),
        "test_evaluated": False,
    }
    definition["campaign_id"] = "csv2-" + stable_hash(definition)[:16]
    return definition


def valid_result(path, manifest, screening, dataset, cap):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = result.get("validation_summary", {})
    finite = []
    for objective, metric in (
        ("self_scaled", "own_rmse"),
        ("fixed_reference", "reference_rmse"),
    ):
        try:
            finite.extend(
                float(value)
                for value in summary[objective][metric]["values"]
            )
        except Exception:
            return False
    return (
        all(value == value and abs(value) < float("inf") for value in finite)
        and result.get("protocol") == "confirmatory-rul-cap-v2"
        and result.get("campaign_id") == manifest["campaign_id"]
        and result.get("dataset") == dataset
        and int(result.get("train_cap")) == int(cap)
        and result.get("reporting_cap") == 125.0
        and result.get("split_seed") == PRIMARY_SPLIT
        and result.get("endpoint_seed") == PRIMARY_SPLIT + 1000003
        and result.get("model_seeds") == [43, 44]
        and result.get("validation_only") is True
        and result.get("test_evaluated") is False
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == screening["trainer_sha256"]
        and result.get("config", {}).get("epochs") == screening["epochs"]
        and result.get("config", {}).get("patience") == screening["patience"]
        and result.get("config", {}).get("train_stride")
        == screening["train_stride"]
        and result.get("config", {}).get("validation_interval")
        == screening["validation_interval"]
        and result.get("config", {}).get("carrier") == screening["carrier"]
    )


def run_additional(args, screening, manifest, dataset, cap):
    tag = f"stability_cap{cap}_split{PRIMARY_SPLIT}_seeds43-44"
    path = args.output_root / "results" / f"{dataset}_{tag}.json"
    if valid_result(path, manifest, screening, dataset, cap):
        print(f"SKIP stability {dataset} cap={cap}", flush=True)
        return path
    if path.exists():
        raise RuntimeError(f"Invalid immutable stability result: {path}")
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
        str(PRIMARY_SPLIT),
        "--endpoint_seed",
        str(PRIMARY_SPLIT + 1000003),
        "--seeds",
        "2",
        "--seed_base",
        "43",
        "--epochs",
        str(screening["epochs"]),
        "--patience",
        str(screening["patience"]),
        "--train_stride",
        str(screening["train_stride"]),
        "--validation_interval",
        str(screening["validation_interval"]),
        "--carrier",
        screening["carrier"],
        "--validation_only",
    ]
    log = (
        args.output_root
        / "stability_logs"
        / f"{dataset}_{tag}_{manifest['campaign_id']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        attempt = 2
        while True:
            candidate = log.with_name(
                f"{log.stem}_attempt{attempt}{log.suffix}"
            )
            if not candidate.exists():
                log = candidate
                break
            attempt += 1
    print(f"RUN stability {dataset} cap={cap}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Stability run failed; see {log}")
    if not valid_result(path, manifest, screening, dataset, cap):
        raise RuntimeError(f"Stability result failed validation: {path}")
    return path


def mean_std(values):
    mean = sum(values) / len(values)
    std = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5
    return mean, std


def build_final_lock(args, screening, screening_lock, stability, paths):
    path_map = {
        (
            json.loads(path.read_text(encoding="utf-8"))["dataset"],
            int(json.loads(path.read_text(encoding="utf-8"))["train_cap"]),
        ): json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    }
    decisions = {}
    for dataset in DATASETS:
        decisions[dataset] = {}
        screening_results = {}
        for cap in CAPS:
            screening_results[cap] = {}
            for split_seed in (20260724, 20260725, 20260726):
                path = (
                    args.output_root
                    / "results"
                    / f"{dataset}_cap{cap}_split{split_seed}.json"
                )
                result = json.loads(path.read_text(encoding="utf-8"))
                if (
                    result.get("campaign_id") != screening["campaign_id"]
                    or result.get("model_seeds") != [42]
                    or result.get("split_seed") != split_seed
                    or result.get("validation_only") is not True
                    or result.get("test_evaluated") is not False
                ):
                    raise RuntimeError(f"Invalid screening source {path}")
                screening_results[cap][split_seed] = result
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            candidates = []
            for cap in CAPS:
                value42 = float(
                    screening_results[cap][PRIMARY_SPLIT][
                        "per_seed_validation"
                    ][0][
                        "checkpoints"
                    ][objective]["metrics"][metric]
                )
                additional = [
                    float(
                        item["checkpoints"][objective]["metrics"][metric]
                    )
                    for item in path_map[(dataset, cap)][
                        "per_seed_validation"
                    ]
                ]
                primary_seed_values = [value42, *additional]
                primary_mean, primary_std = mean_std(
                    primary_seed_values
                )
                split_values = [primary_mean]
                split_records = [
                    {
                        "split_seed": PRIMARY_SPLIT,
                        "model_seed_values": primary_seed_values,
                        "split_mean": primary_mean,
                        "model_seed_std": primary_std,
                    }
                ]
                for split_seed in (20260725, 20260726):
                    split_value = float(
                        screening_results[cap][split_seed][
                            "per_seed_validation"
                        ][0]["checkpoints"][objective]["metrics"][metric]
                    )
                    split_values.append(split_value)
                    split_records.append(
                        {
                            "split_seed": split_seed,
                            "model_seed_values": [split_value],
                            "split_mean": split_value,
                            "model_seed_std": 0.0,
                        }
                    )
                mean, std = mean_std(split_values)
                candidates.append(
                    {
                        "cap": cap,
                        "split_records": split_records,
                        "split_values": split_values,
                        "mean": mean,
                        "std_across_splits": std,
                    }
                )
            selected = min(
                candidates, key=lambda item: (item["mean"], -item["cap"])
            )
            decisions[dataset][objective] = {
                "metric": metric,
                "selected_cap": selected["cap"],
                "selection_value": selected["mean"],
                "candidate_table": candidates,
                "screening_split_choices": screening_lock["decisions"][
                    dataset
                ][objective]["split_choices"],
            }
        fixed_candidate = next(
            item
            for item in decisions[dataset]["fixed_reference"][
                "candidate_table"
            ]
            if item["cap"] == 125
        )
        decisions[dataset]["fixed_cap_125"] = {
            "metric": "reference_rmse",
            "selected_cap": 125,
            "selection_value": fixed_candidate["mean"],
            "candidate_table": [fixed_candidate],
            "role": "predeclared fixed-cap control",
        }
    final_lock = {
        "protocol": "confirmatory-final-cap-lock-v2",
        "screening_campaign_id": screening["campaign_id"],
        "stability_campaign_id": stability["campaign_id"],
        "screening_lock_sha256": stable_hash(screening_lock),
        "stability_manifest_sha256": stable_hash(stability),
        "created_unix": time.time(),
        "selection_rule": stability["selection_rule"],
        "decisions": decisions,
        "test_evaluated": False,
    }
    lock_path = args.output_root / "final_cap_decisions.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("decisions") != decisions:
            raise RuntimeError("Existing final cap lock differs")
    else:
        exclusive_json(lock_path, final_lock)
    return lock_path, final_lock


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    screening_path = args.output_root / "campaign_manifest.json"
    screening_lock_path = args.output_root / "locked_cap_decisions.json"
    if not screening_path.exists() or not screening_lock_path.exists():
        raise RuntimeError("Three-split screening campaign is not locked")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    screening_lock = json.loads(
        screening_lock_path.read_text(encoding="utf-8")
    )
    if screening.get("trainer_sha256") != sha256_file(TRAINER):
        raise RuntimeError("Trainer changed after the screening campaign")
    stability = stability_manifest(screening)
    manifest_path = args.output_root / "stability_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != stability:
            raise RuntimeError("Existing stability manifest differs")
    else:
        exclusive_json(manifest_path, stability)

    paths = []
    for dataset in DATASETS:
        for cap in CAPS:
            paths.append(
                run_additional(
                    args, screening, stability, dataset, cap
                )
            )
    lock_path, final_lock = build_final_lock(
        args, screening, screening_lock, stability, paths
    )
    summary = {
        "status": "model_seed_stability_complete",
        "stability_campaign_id": stability["campaign_id"],
        "completed_runs": len(paths),
        "final_lock_path": str(lock_path),
        "decisions": final_lock["decisions"],
        "test_evaluated": False,
    }
    atomic_json(args.output_root / "stability_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
