"""Freeze the observed cap-screen interruption/recovery provenance.

This script records only already-completed facts. It never launches training,
opens C-MAPSS test/truth files, or changes a selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "confirmatory_rul_train.py"
CAP_RUNNER = HERE / "run_confirmatory_cap_sweep.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Missing selection-attempt evidence: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def validate_validation_result(
    path: Path,
    *,
    campaign_id: str,
    dataset: str,
    cap: int,
    split_seed: int,
) -> None:
    result = load_json(path)
    if (
        result.get("campaign_id") != campaign_id
        or result.get("dataset") != dataset
        or int(result.get("train_cap", -1)) != cap
        or int(result.get("split_seed", -1)) != split_seed
        or result.get("validation_only") is not True
        or result.get("test_evaluated") is not False
        or result.get("model_seeds") != [42]
        or result.get("source_fingerprints", {}).get("trainer_sha256")
        != sha256_file(TRAINER)
        or "test_results" in result
        or any(
            key in result.get("data_fingerprints", {})
            for key in ("test_sha256", "truth_sha256")
        )
    ):
        raise RuntimeError(
            f"Recovered selection result is not the frozen validation cell: {path}"
        )


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    campaign = load_json(root / "campaign_manifest.json")
    campaign_id = campaign.get("campaign_id")
    if (
        campaign.get("protocol") != "confirmatory-cap-sweep-v2"
        or campaign.get("test_evaluated") is not False
        or not campaign_id
    ):
        raise RuntimeError("Campaign manifest is invalid")

    evidence_relatives = [
        "logs/orchestration_notes.log",
        (
            "logs/FD004_cap100_split20260724_"
            f"{campaign_id}.log"
        ),
        (
            "logs/FD004_cap100_split20260724_"
            f"{campaign_id}_attempt2.log"
        ),
        (
            "logs/FD004_cap115_split20260726_"
            f"{campaign_id}_parallel.stdout.log"
        ),
        (
            "logs/FD004_cap115_split20260726_"
            f"{campaign_id}_parallel.stderr.log"
        ),
        (
            "logs/FD004_cap125_split20260726_"
            f"{campaign_id}_parallel.stdout.log"
        ),
        (
            "logs/FD004_cap125_split20260726_"
            f"{campaign_id}_parallel.stderr.log"
        ),
        "parallel_resume_stderr.log",
        "parallel_resume_stdout.log",
        "parallel_resume_v2_stderr.log",
        "parallel_resume_v2_stdout.log",
        "results/FD004_cap100_split20260724.json",
        "results/FD004_cap115_split20260726.json",
        "results/FD004_cap125_split20260726.json",
        "cap_sweep_summary.json",
        "final_cap_decisions.json",
    ]
    evidence = {
        relative: {
            "bytes": require_file(root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in evidence_relatives
    }

    original_log = require_file(
        root
        / "logs"
        / f"FD004_cap100_split20260724_{campaign_id}.log"
    ).read_text(encoding="utf-8", errors="replace")
    retry_log = require_file(
        root
        / "logs"
        / f"FD004_cap100_split20260724_{campaign_id}_attempt2.log"
    ).read_text(encoding="utf-8", errors="replace")
    result_path = root / "results" / "FD004_cap100_split20260724.json"
    expected_write = f"WROTE {result_path}"
    if expected_write in original_log or expected_write not in retry_log:
        raise RuntimeError(
            "FD004 cap100 interruption/retry log sequence is inconsistent"
        )
    validate_validation_result(
        result_path,
        campaign_id=campaign_id,
        dataset="FD004",
        cap=100,
        split_seed=20260724,
    )
    for cap in (115, 125):
        validate_validation_result(
            root / "results" / f"FD004_cap{cap}_split20260726.json",
            campaign_id=campaign_id,
            dataset="FD004",
            cap=cap,
            split_seed=20260726,
        )

    payload = {
        "protocol": "selection-attempt-provenance-v1",
        "campaign_id": campaign_id,
        "selection_complete_before_manifest": True,
        "selection_changed_by_manifest": False,
        "test_evaluated": False,
        "documented_interruption": {
            "dataset": "FD004",
            "cap": 100,
            "split_seed": 20260724,
            "first_attempt_result_written": False,
            "second_attempt_result_written": True,
            "reason": (
                "the serial parent was replaced after epoch 20 before any "
                "result JSON was written"
            ),
            "configuration_changed": False,
        },
        "independently_launched_cells": [
            {"dataset": "FD004", "cap": 115, "split_seed": 20260726},
            {"dataset": "FD004", "cap": 125, "split_seed": 20260726},
        ],
        "orchestration_helper_source_archived": False,
        "provenance_limitation": (
            "The temporary parallel orchestration helper source was not "
            "retained. Its stdout/stderr, cell logs, canonical runner source, "
            "result payloads, and final decision files are hash-bound here."
        ),
        "source_sha256": {
            "manifest_builder": sha256_file(Path(__file__).resolve()),
            "canonical_cap_runner": sha256_file(CAP_RUNNER),
            "frozen_trainer": sha256_file(TRAINER),
        },
        "evidence_files": evidence,
    }
    payload["manifest_sha256_without_self"] = stable_hash(payload)
    return payload


def write_create_or_validate(path: Path, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "Existing selection-attempt manifest differs from evidence"
            )
        return
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    args = parser.parse_args()
    payload = build_manifest(args.output_root)
    output = args.output_root.resolve() / "SELECTION_ATTEMPT_MANIFEST.json"
    write_create_or_validate(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
