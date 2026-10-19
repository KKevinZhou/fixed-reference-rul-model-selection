"""Validation-only sensitivity check on a second, pre-existing neural carrier.

The check is deliberately separated from the standard C-MAPSS test campaign.
It uses only grouped training-data validation endpoints and cannot change any
cap or hyperparameter decision consumed by ``run_confirmatory_final.py``.
"""

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


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "confirmatory_rul_train.py"
MODEL = HERE / "baselines" / "models_v3.py"
DATASETS = ("FD001", "FD002")
CAPS = (90, 100, 110, 115, 125)
SPLIT_SEED = 20260724
ENDPOINT_SEED = SPLIT_SEED + 1000003
MODEL_SEED = 42
CONFIG = {
    "FD001": {"seq_len": 50, "lr": 5e-4},
    "FD002": {"seq_len": 50, "lr": 1e-3},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def valid_result(
    path: Path,
    *,
    campaign_id: str,
    dataset: str,
    cap: int,
    source_manifest: dict,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        metrics = result["validation_summary"]
        values = (
            metrics["self_scaled"]["own_rmse"]["values"]
            + metrics["fixed_reference"]["reference_rmse"]["values"]
        )
    except Exception:
        return False
    return (
        result.get("campaign_id") == campaign_id
        and result.get("dataset") == dataset
        and float(result.get("train_cap")) == float(cap)
        and result.get("reporting_cap") == 125.0
        and result.get("split_seed") == SPLIT_SEED
        and result.get("endpoint_seed") == ENDPOINT_SEED
        and result.get("model_seeds") == [MODEL_SEED]
        and result.get("validation_only") is True
        and result.get("test_evaluated") is False
        and result.get("config", {}).get("carrier") == "dualstream"
        and int(config["seq_len"]) == CONFIG[dataset]["seq_len"]
        and math.isclose(float(config["lr"]), CONFIG[dataset]["lr"])
        and int(config["batch"]) == 128
        and math.isclose(float(config["huber_delta"]), 1.0)
        and all(math.isfinite(float(value)) for value in values)
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == source_manifest["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == source_manifest["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == source_manifest["raw_train_sha256"][dataset]
        and "test_sha256" not in result.get("data_fingerprints", {})
        and "truth_sha256" not in result.get("data_fingerprints", {})
    )


def run_one(
    args: argparse.Namespace,
    plan: dict,
    source_manifest: dict,
    dataset: str,
    cap: int,
) -> Path:
    if (
        sha256_file(TRAINER) != plan["trainer_sha256"]
        or sha256_file(MODEL) != plan["model_sha256"]
    ):
        raise RuntimeError("Frozen trainer or model changed during carrier check")
    tag = f"dualstream_transfer_cap{cap}_split{SPLIT_SEED}_seed{MODEL_SEED}"
    result_path = args.transfer_root / "results" / f"{dataset}_{tag}.json"
    if valid_result(
        result_path,
        campaign_id=plan["campaign_id"],
        dataset=dataset,
        cap=cap,
        source_manifest=source_manifest,
    ):
        print(f"SKIP {dataset} cap={cap}", flush=True)
        return result_path
    if result_path.exists():
        raise RuntimeError(f"Invalid immutable transfer result: {result_path}")
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--dataset",
        dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.transfer_root),
        "--campaign_id",
        plan["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(cap),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(SPLIT_SEED),
        "--endpoint_seed",
        str(ENDPOINT_SEED),
        "--seeds",
        "1",
        "--seed_base",
        str(MODEL_SEED),
        "--seq_len",
        str(CONFIG[dataset]["seq_len"]),
        "--lr",
        str(CONFIG[dataset]["lr"]),
        "--batch",
        "128",
        "--huber_delta",
        "1.0",
        "--epochs",
        str(plan["epochs"]),
        "--patience",
        str(plan["patience"]),
        "--train_stride",
        str(plan["train_stride"]),
        "--validation_interval",
        str(plan["validation_interval"]),
        "--carrier",
        "dualstream",
        "--validation_only",
    ]
    log = (
        args.transfer_root
        / "logs"
        / f"{dataset}_cap{cap}_{plan['campaign_id']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {dataset} cap={cap}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=HERE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Carrier-transfer run failed; see {log}")
    if not valid_result(
        result_path,
        campaign_id=plan["campaign_id"],
        dataset=dataset,
        cap=cap,
        source_manifest=source_manifest,
    ):
        raise RuntimeError(f"Carrier-transfer result failed validation: {result_path}")
    return result_path


def selected_cap(profile: list[dict], metric: str) -> int:
    return int(
        min(
            profile,
            key=lambda row: (float(row[metric]), -int(row["cap"])),
        )["cap"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--campaign_root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise RuntimeError("Carrier-transfer workers must be one or two")
    args.data_root = args.data_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.transfer_root = args.campaign_root / "carrier_transfer"
    args.transfer_root.mkdir(parents=True, exist_ok=True)
    for forbidden in (
        "TEST_EVALUATION_STARTED.json",
        "TEST_DATA_VERIFIED.json",
        "TEST_EVALUATED",
        "final_test_summary.json",
    ):
        if (args.campaign_root / forbidden).exists():
            raise RuntimeError(
                "Carrier-sensitivity validation must finish before test "
                f"state exists: {forbidden}"
            )
    source_manifest = json.loads(
        (args.campaign_root / "campaign_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    trainer_hash = sha256_file(TRAINER)
    model_hash = sha256_file(MODEL)
    if (
        trainer_hash != source_manifest.get("trainer_sha256")
        or model_hash != source_manifest.get("model_sha256")
    ):
        raise RuntimeError("Carrier-transfer source differs from frozen campaign")
    plan = {
        "protocol": "confirmatory-carrier-transfer-v1",
        "role": (
            "validation-only single-split, single-seed carrier sensitivity "
            "check; cannot change the locked cap-grid, TPE, or standard-test "
            "reporting decisions"
        ),
        "datasets": list(DATASETS),
        "caps": list(CAPS),
        "reporting_cap": 125,
        "split_seed": SPLIT_SEED,
        "endpoint_seed": ENDPOINT_SEED,
        "model_seed": MODEL_SEED,
        "carrier": "dualstream",
        "config": CONFIG,
        "epochs": int(source_manifest["epochs"]),
        "patience": int(source_manifest["patience"]),
        "train_stride": int(source_manifest["train_stride"]),
        "validation_interval": int(source_manifest["validation_interval"]),
        "selection_rule": (
            "minimum primary-split validation objective; exact ties prefer "
            "the larger cap"
        ),
        "primary_campaign_manifest_sha256": sha256_file(
            args.campaign_root / "campaign_manifest.json"
        ),
        "tpe_robustness_summary_sha256": sha256_file(
            args.campaign_root / "tpe_robustness_summary.json"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": trainer_hash,
        "model_sha256": model_hash,
        "raw_train_sha256": {
            dataset: source_manifest["raw_train_sha256"][dataset]
            for dataset in DATASETS
        },
        "test_files_permitted": False,
    }
    plan["campaign_id"] = "carrierv1-" + stable_hash(plan)[:16]
    plan_path = args.campaign_root / "carrier_transfer_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Existing carrier-transfer plan differs")
    else:
        exclusive_json(plan_path, plan)
    tasks = [(dataset, cap) for dataset in DATASETS for cap in CAPS]
    paths: dict[tuple[str, int], Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                args,
                plan,
                source_manifest,
                dataset,
                cap,
            ): (dataset, cap)
            for dataset, cap in tasks
        }
        for future in as_completed(futures):
            paths[futures[future]] = future.result()
    if set(paths) != set(tasks):
        raise RuntimeError("Carrier-transfer scheduler did not complete all runs")
    selections = {}
    result_sha256 = {}
    for dataset in DATASETS:
        profile = []
        for cap in CAPS:
            path = paths[(dataset, cap)]
            result = json.loads(path.read_text(encoding="utf-8"))
            profile.append(
                {
                    "cap": cap,
                    "self_scaled_own_rmse": float(
                        result["validation_summary"]["self_scaled"][
                            "own_rmse"
                        ]["mean"]
                    ),
                    "self_checkpoint_reference_rmse": float(
                        result["validation_summary"]["self_scaled"][
                            "reference_rmse"
                        ]["mean"]
                    ),
                    "fixed_reference_rmse": float(
                        result["validation_summary"]["fixed_reference"][
                            "reference_rmse"
                        ]["mean"]
                    ),
                }
            )
            result_sha256[str(path)] = sha256_file(path)
        self_cap = selected_cap(profile, "self_scaled_own_rmse")
        fixed_cap = selected_cap(profile, "fixed_reference_rmse")
        selections[dataset] = {
            "self_scaled_selected_cap": self_cap,
            "fixed_reference_selected_cap": fixed_cap,
            "selection_changed": self_cap != fixed_cap,
            "profile": profile,
        }
    summary = {
        "status": "complete",
        "protocol": plan["protocol"],
        "campaign_id": plan["campaign_id"],
        "plan_sha256": sha256_file(plan_path),
        "selection_frozen": True,
        "selection_changed_count": sum(
            int(value["selection_changed"]) for value in selections.values()
        ),
        "selections": selections,
        "result_sha256": result_sha256,
        "test_evaluated": False,
    }
    atomic_json(args.campaign_root / "carrier_transfer_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
