"""One-shot, exact-configuration recovery for failed carrier-transfer cells.

This helper is intentionally separate from the frozen carrier-transfer runner.
It may be used only before the standard-test start marker exists, after the
original carrier-transfer process has stopped, and only for cells whose
original log records a non-finite training loss.  Every retry uses the exact
plan-bound command, runs sequentially, retains the failed log, and writes a
hash-bound audit record.  No hyperparameter or seed may change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import run_carrier_transfer as carrier


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "confirmatory_rul_train.py"
MODEL = HERE / "baselines" / "models_v3.py"
FAILURE_SIGNATURE = "FloatingPointError: Non-finite training loss"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


def atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def retry_command(
    *,
    data_root: Path,
    transfer_root: Path,
    plan: dict,
    dataset: str,
    cap: int,
) -> list[str]:
    tag = (
        f"dualstream_transfer_cap{cap}_"
        f"split{carrier.SPLIT_SEED}_seed{carrier.MODEL_SEED}"
    )
    return [
        sys.executable,
        "-u",
        str(TRAINER),
        "--dataset",
        dataset,
        "--data_root",
        str(data_root),
        "--output_root",
        str(transfer_root),
        "--campaign_id",
        plan["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(cap),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(carrier.SPLIT_SEED),
        "--endpoint_seed",
        str(carrier.ENDPOINT_SEED),
        "--seeds",
        "1",
        "--seed_base",
        str(carrier.MODEL_SEED),
        "--seq_len",
        str(carrier.CONFIG[dataset]["seq_len"]),
        "--lr",
        str(carrier.CONFIG[dataset]["lr"]),
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


def result_path(transfer_root: Path, dataset: str, cap: int) -> Path:
    tag = (
        f"dualstream_transfer_cap{cap}_"
        f"split{carrier.SPLIT_SEED}_seed{carrier.MODEL_SEED}"
    )
    return transfer_root / "results" / f"{dataset}_{tag}.json"


def original_log(
    transfer_root: Path, plan: dict, dataset: str, cap: int
) -> Path:
    return (
        transfer_root
        / "logs"
        / f"{dataset}_cap{cap}_{plan['campaign_id']}.log"
    )


def retry_log(
    transfer_root: Path, plan: dict, dataset: str, cap: int
) -> Path:
    return (
        transfer_root
        / "logs"
        / f"{dataset}_cap{cap}_{plan['campaign_id']}_retry1.log"
    )


def discover_failed_cells(
    *,
    data_root: Path,
    transfer_root: Path,
    plan: dict,
    source_manifest: dict,
) -> list[dict]:
    cells = []
    for dataset in carrier.DATASETS:
        for cap in carrier.CAPS:
            path = result_path(transfer_root, dataset, cap)
            if carrier.valid_result(
                path,
                campaign_id=plan["campaign_id"],
                dataset=dataset,
                cap=cap,
                source_manifest=source_manifest,
            ):
                continue
            if path.exists():
                raise RuntimeError(f"Invalid immutable result exists: {path}")
            log = original_log(transfer_root, plan, dataset, cap)
            if not log.is_file():
                raise RuntimeError(
                    f"Missing result has no original attempt log: {path}"
                )
            text = log.read_text(encoding="utf-8", errors="replace")
            if FAILURE_SIGNATURE not in text:
                raise RuntimeError(
                    "Retry is permitted only for an explicitly recorded "
                    f"non-finite loss: {log}"
                )
            command = retry_command(
                data_root=data_root,
                transfer_root=transfer_root,
                plan=plan,
                dataset=dataset,
                cap=cap,
            )
            cells.append(
                {
                    "dataset": dataset,
                    "cap": cap,
                    "result_path": str(path),
                    "original_log": str(log),
                    "original_log_sha256": sha256_file(log),
                    "failure_signature": FAILURE_SIGNATURE,
                    "command_tail": command[3:],
                }
            )
    return cells


def terminal_carrier_summary(
    *,
    campaign_root: Path,
    transfer_root: Path,
    plan: dict,
    source_manifest: dict,
    retry_summary_path: Path,
    outcomes: list[dict],
) -> dict:
    """Freeze a transparent partial profile after an exact retry still fails."""
    result_hashes = {}
    profiles = {}
    selections = {}
    missing = []
    for dataset in carrier.DATASETS:
        profile = []
        for cap in carrier.CAPS:
            path = result_path(transfer_root, dataset, cap)
            if not carrier.valid_result(
                path,
                campaign_id=plan["campaign_id"],
                dataset=dataset,
                cap=cap,
                source_manifest=source_manifest,
            ):
                missing.append({"dataset": dataset, "cap": cap})
                continue
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
            result_hashes[str(path)] = sha256_file(path)
        profiles[dataset] = profile
        if len(profile) == len(carrier.CAPS):
            self_cap = carrier.selected_cap(
                profile, "self_scaled_own_rmse"
            )
            fixed_cap = carrier.selected_cap(
                profile, "fixed_reference_rmse"
            )
            selections[dataset] = {
                "self_scaled_selected_cap": self_cap,
                "fixed_reference_selected_cap": fixed_cap,
                "selection_changed": self_cap != fixed_cap,
                "profile": profile,
            }
    if not missing:
        raise RuntimeError(
            "Terminal carrier summary requested without a missing cell"
        )
    failed_outcomes = [
        {
            "dataset": outcome["dataset"],
            "cap": int(outcome["cap"]),
            "returncode": int(outcome["returncode"]),
            "result_valid": bool(outcome["result_valid"]),
            "failure_signature_recurred": bool(
                outcome["failure_signature_recurred"]
            ),
            "retry_log": outcome["retry_log"],
            "retry_log_sha256": outcome["retry_log_sha256"],
        }
        for outcome in outcomes
        if not outcome["result_valid"]
    ]
    return {
        "status": "terminal_failure",
        "protocol": plan["protocol"],
        "campaign_id": plan["campaign_id"],
        "plan_sha256": sha256_file(
            campaign_root / "carrier_transfer_plan.json"
        ),
        "selection_frozen": True,
        "selection_changed_count": sum(
            int(value["selection_changed"])
            for value in selections.values()
        ),
        "selections": selections,
        "partial_profiles": profiles,
        "missing_cells": missing,
        "failed_retry_outcomes": failed_outcomes,
        "result_sha256": result_hashes,
        "retry_summary_sha256": sha256_file(retry_summary_path),
        "terminal_reason": (
            "at least one plan-bound cell retained non-finite training "
            "loss after one exact same-configuration sequential retry"
        ),
        "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--campaign_root", type=Path, required=True)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    transfer_root = args.campaign_root / "carrier_transfer"

    for forbidden in (
        "TEST_EVALUATION_STARTED.json",
        "TEST_DATA_VERIFIED.json",
        "TEST_EVALUATED",
        "final_test_summary.json",
    ):
        if (args.campaign_root / forbidden).exists():
            raise RuntimeError(
                "Carrier retry is forbidden after standard-test state exists: "
                f"{forbidden}"
            )
    plan_path = args.campaign_root / "carrier_transfer_plan.json"
    source_path = args.campaign_root / "campaign_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        sha256_file(TRAINER) != plan["trainer_sha256"]
        or sha256_file(MODEL) != plan["model_sha256"]
        or plan.get("test_files_permitted") is not False
    ):
        raise RuntimeError("Frozen carrier sources or plan boundary changed")

    cells = discover_failed_cells(
        data_root=args.data_root,
        transfer_root=transfer_root,
        plan=plan,
        source_manifest=source_manifest,
    )
    if not cells:
        raise RuntimeError("No eligible failed carrier-transfer cells found")
    manifest_path = args.campaign_root / "carrier_transfer_retry_manifest.json"
    manifest = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": plan["campaign_id"],
        "carrier_plan_sha256": sha256_file(plan_path),
        "retry_helper_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(TRAINER),
        "model_sha256": sha256_file(MODEL),
        "failure_cause": "non-finite training loss",
        "attempt_limit_per_cell": 1,
        "execution_mode": "sequential",
        "parameter_changes": False,
        "test_files_permitted": False,
        "cells": cells,
        "created_unix": time.time(),
    }
    exclusive_json(manifest_path, manifest)

    outcomes = []
    failed = False
    for cell in cells:
        dataset = cell["dataset"]
        cap = int(cell["cap"])
        command = retry_command(
            data_root=args.data_root,
            transfer_root=transfer_root,
            plan=plan,
            dataset=dataset,
            cap=cap,
        )
        log = retry_log(transfer_root, plan, dataset, cap)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=HERE,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        path = result_path(transfer_root, dataset, cap)
        valid = carrier.valid_result(
            path,
            campaign_id=plan["campaign_id"],
            dataset=dataset,
            cap=cap,
            source_manifest=source_manifest,
        )
        retry_text = log.read_text(encoding="utf-8", errors="replace")
        outcome = {
            "dataset": dataset,
            "cap": cap,
            "returncode": process.returncode,
            "retry_log": str(log),
            "retry_log_sha256": sha256_file(log),
            "result_path": str(path),
            "result_valid": valid,
            "failure_signature_recurred": (
                FAILURE_SIGNATURE in retry_text
            ),
        }
        if valid:
            outcome["result_sha256"] = sha256_file(path)
        else:
            failed = True
        outcomes.append(outcome)

    summary_path = args.campaign_root / "carrier_transfer_retry_summary.json"
    summary = {
        "protocol": manifest["protocol"],
        "campaign_id": plan["campaign_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "status": "failed" if failed else "complete",
        "outcomes": outcomes,
        "test_evaluated": False,
    }
    atomic_json(summary_path, summary)
    if failed:
        if any(
            not outcome["result_valid"]
            and not outcome["failure_signature_recurred"]
            for outcome in outcomes
        ):
            raise RuntimeError(
                "Exact retry ended without a valid result but did not "
                "reproduce the classified non-finite-loss signature"
            )
        terminal = terminal_carrier_summary(
            campaign_root=args.campaign_root,
            transfer_root=transfer_root,
            plan=plan,
            source_manifest=source_manifest,
            retry_summary_path=summary_path,
            outcomes=outcomes,
        )
        atomic_json(
            args.campaign_root / "carrier_transfer_summary.json",
            terminal,
        )
        raise RuntimeError(
            "Exact carrier retry failed; retain the failed cell as evidence"
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_carrier_transfer.py"),
            "--data_root",
            str(args.data_root),
            "--campaign_root",
            str(args.campaign_root),
            "--workers",
            "1",
        ],
        cwd=HERE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Frozen carrier runner could not finalize retries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
