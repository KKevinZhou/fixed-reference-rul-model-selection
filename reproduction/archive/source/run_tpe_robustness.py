"""Evaluate locked TPE configurations on splits held out from TPE optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "confirmatory_rul_train.py"
MODEL = SCRIPT_DIR / "baselines" / "models_v3.py"
STUDIES = [
    ("FD001", "self_scaled"),
    ("FD001", "fixed_reference"),
    ("FD001", "fixed_cap_125"),
    ("FD002", "self_scaled"),
    ("FD002", "fixed_reference"),
    ("FD002", "fixed_cap_125"),
]
DEFAULT_SEQ = {"FD001": 50, "FD002": 50}


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, value):
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
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def valid(
    path,
    campaign_id,
    dataset,
    params,
    split_seed,
    seeds,
    screening,
):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        metrics = result["validation_summary"]
        finite = all(
            math.isfinite(float(value))
            for objective, metric in (
                ("self_scaled", "own_rmse"),
                ("fixed_reference", "reference_rmse"),
            )
            for value in metrics[objective][metric]["values"]
        )
    except Exception:
        return False
    return (
        finite
        and result.get("campaign_id") == campaign_id
        and result.get("dataset") == dataset
        and result.get("validation_only") is True
        and result.get("test_evaluated") is False
        and result.get("split_seed") == split_seed
        and result.get("endpoint_seed") == split_seed + 1000003
        and result.get("model_seeds") == seeds
        and int(result.get("train_cap")) == int(params["train_cap"])
        and math.isclose(float(config["lr"]), float(params["lr"]))
        and int(config["batch"]) == int(params["batch"])
        and math.isclose(
            float(config["huber_delta"]), float(params["huber_delta"])
        )
        and int(config["seq_len"]) == int(DEFAULT_SEQ[dataset])
        and int(config["epochs"]) == int(screening["epochs"])
        and int(config["patience"]) == int(screening["patience"])
        and int(config["train_stride"]) == int(screening["train_stride"])
        and int(config["validation_interval"])
        == int(screening["validation_interval"])
        and config["carrier"] == screening["carrier"]
        and result.get("reporting_cap") == 125.0
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == screening["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == screening["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == screening["raw_train_sha256"][dataset]
        and "test_sha256" not in result.get("data_fingerprints", {})
        and "truth_sha256" not in result.get("data_fingerprints", {})
    )


def run_one(
    args,
    screening,
    campaign_id,
    dataset,
    objective,
    params,
    split_seed,
    seed_base,
    seeds,
):
    if (
        sha256_file(TRAINER) != screening["trainer_sha256"]
        or sha256_file(MODEL) != screening["model_sha256"]
    ):
        raise RuntimeError(
            "Training source changed during robustness execution"
        )
    seed_label = f"s{seed_base}-{seed_base + seeds - 1}"
    tag = (
        f"tpe_robust_{objective}_split{split_seed}_{seed_label}_"
        f"{stable_hash(params)[:8]}"
    )
    path = args.output_root / "results" / f"{dataset}_{tag}.json"
    expected_seeds = list(range(seed_base, seed_base + seeds))
    if valid(
        path,
        campaign_id,
        dataset,
        params,
        split_seed,
        expected_seeds,
        screening,
    ):
        print(f"SKIP {dataset} {objective} {split_seed} {seed_label}", flush=True)
        return path
    if path.exists():
        raise RuntimeError(f"Invalid immutable robustness result: {path}")
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
        campaign_id,
        "--tag",
        tag,
        "--train_cap",
        str(params["train_cap"]),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(split_seed),
        "--endpoint_seed",
        str(split_seed + 1000003),
        "--seeds",
        str(seeds),
        "--seed_base",
        str(seed_base),
        "--seq_len",
        str(DEFAULT_SEQ[dataset]),
        "--lr",
        str(params["lr"]),
        "--batch",
        str(params["batch"]),
        "--huber_delta",
        str(params["huber_delta"]),
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
        / "tpe_robustness_logs"
        / f"{dataset}_{tag}_{campaign_id}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {dataset} {objective} {split_seed} {seed_label}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Robustness run failed; see {log}")
    if not valid(
        path,
        campaign_id,
        dataset,
        params,
        split_seed,
        expected_seeds,
        screening,
    ):
        raise RuntimeError(f"Robustness result failed validation: {path}")
    return path


def main():
    args = parse_args()
    if not 1 <= args.workers <= 3:
        raise RuntimeError("Robustness workers must be between 1 and 3")
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    screening = json.loads(
        (args.output_root / "campaign_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        sha256_file(TRAINER) != screening.get("trainer_sha256")
        or sha256_file(MODEL) != screening.get("model_sha256")
    ):
        raise RuntimeError("Trainer or model source changed after screening")
    locks = {}
    for dataset, objective in STUDIES:
        path = (
            args.output_root
            / "tpe_locks"
            / f"{dataset}_{objective}.json"
        )
        if not path.exists():
            raise RuntimeError(f"Missing TPE lock {path}")
        locks[(dataset, objective)] = json.loads(
            path.read_text(encoding="utf-8")
        )
    definition = {
        "protocol": "confirmatory-tpe-robustness-v2",
        "screening_campaign_id": screening["campaign_id"],
        "tpe_lock_hashes": {
            f"{dataset}_{objective}": stable_hash(locks[(dataset, objective)])
            for dataset, objective in STUDIES
        },
        "secondary_split_seeds": [20260725, 20260726],
        "primary_additional_model_seeds": [43, 44],
        "workers": args.workers,
        "robustness_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "trainer_sha256": screening["trainer_sha256"],
        "model_sha256": screening["model_sha256"],
        "selection_frozen": True,
        "role": (
            "report-only post-lock internal robustness evaluation; "
            "does not change the selected trial"
        ),
        "test_evaluated": False,
    }
    definition["campaign_id"] = "tperobv2-" + stable_hash(definition)[:16]
    manifest_path = args.output_root / "tpe_robustness_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != definition:
            raise RuntimeError("Existing TPE robustness manifest differs")
    else:
        exclusive_json(manifest_path, definition)

    result_paths = {}
    tasks = []
    for dataset, objective in STUDIES:
        params = locks[(dataset, objective)]["best_params"]
        key = f"{dataset}_{objective}"
        result_paths[key] = [None, None, None]
        tasks.append(
            (
                key,
                0,
                dataset,
                objective,
                params,
                20260724,
                43,
                2,
            )
        )
        for index, split_seed in enumerate(
            (20260725, 20260726), start=1
        ):
            tasks.append(
                (
                    key,
                    index,
                    dataset,
                    objective,
                    params,
                    split_seed,
                    42,
                    1,
                )
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                args,
                screening,
                definition["campaign_id"],
                dataset,
                objective,
                params,
                split_seed,
                seed_base,
                seeds,
            ): (key, index)
            for (
                key,
                index,
                dataset,
                objective,
                params,
                split_seed,
                seed_base,
                seeds,
            ) in tasks
        }
        for future in as_completed(futures):
            key, index = futures[future]
            result_paths[key][index] = str(future.result())

    if any(
        path is None
        for paths in result_paths.values()
        for path in paths
    ):
        raise RuntimeError("Robustness scheduler did not complete every task")
    result_hashes = {}
    metric_summary = {}
    for key, paths in result_paths.items():
        result_hashes[key] = {}
        metric_summary[key] = []
        objective = key.split("_", 1)[1]
        checkpoint_objective, metric = (
            ("self_scaled", "own_rmse")
            if objective == "self_scaled"
            else ("fixed_reference", "reference_rmse")
        )
        for value in paths:
            path = Path(value)
            result = json.loads(path.read_text(encoding="utf-8"))
            result_hashes[key][str(path)] = sha256_file(path)
            metric_summary[key].append(
                {
                    "split_seed": result["split_seed"],
                    "model_seeds": result["model_seeds"],
                    "selection_metric": metric,
                    "values": result["validation_summary"][
                        checkpoint_objective
                    ][metric]["values"],
                }
            )
    summary = {
        "status": "complete",
        "campaign_id": definition["campaign_id"],
        "manifest_sha256": stable_hash(definition),
        "tpe_lock_hashes": definition["tpe_lock_hashes"],
        "selection_frozen": True,
        "selection_changed": False,
        "role": definition["role"],
        "result_paths": result_paths,
        "result_sha256": result_hashes,
        "metric_summary": metric_summary,
        "test_evaluated": False,
    }
    atomic_json(args.output_root / "tpe_robustness_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
