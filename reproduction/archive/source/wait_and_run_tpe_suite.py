"""Wait for seed stability, then run six equal-trial-budget TPE studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TPE = SCRIPT_DIR / "run_confirmatory_tpe.py"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()
    if args.trials != 12:
        raise RuntimeError("The confirmatory TPE budget is locked to 12 trials")
    if not 1 <= args.workers <= 3:
        raise RuntimeError("TPE study workers must be between 1 and 3")
    marker = args.output_root / "stability_summary.json"
    while not marker.exists():
        print(f"WAIT {marker}", flush=True)
        time.sleep(args.poll_seconds)

    studies = [
        (dataset, objective)
        for dataset in ("FD001", "FD002")
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        )
    ]

    def run_study(dataset, objective):
        command = [
            sys.executable,
            "-u",
            str(TPE),
            "--dataset",
            dataset,
            "--objective",
            objective,
            "--trials",
            str(args.trials),
            "--data_root",
            str(args.data_root),
            "--output_root",
            str(args.output_root),
        ]
        print(f"START TPE {dataset} {objective}", flush=True)
        process = subprocess.run(command, cwd=SCRIPT_DIR)
        if process.returncode != 0:
            raise RuntimeError(
                f"TPE study failed: {dataset} {objective} "
                f"(return code {process.returncode})"
            )
        print(f"DONE TPE {dataset} {objective}", flush=True)
        return dataset, objective

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_study, dataset, objective): (
                dataset,
                objective,
            )
            for dataset, objective in studies
        }
        for future in as_completed(futures):
            future.result()
    lock_hashes = {}
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            path = (
                args.output_root
                / "tpe_locks"
                / f"{dataset}_{objective}.json"
            )
            lock_hashes[f"{dataset}_{objective}"] = sha256_file(path)
    completion = args.output_root / "TPE_SUITE_COMPLETE"
    payload = {
        "status": "complete",
        "studies": 6,
        "trials_per_study": 12,
        "study_workers": args.workers,
        "lock_file_sha256": lock_hashes,
        "test_evaluated": False,
    }
    with completion.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
