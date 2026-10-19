"""Queue the validation-only RUL-cap experiment suite on one GPU.

The queue is resumable: a completed JSON result is never overwritten.  Each
cap is trained twice because checkpoint selection is part of the model-
selection objective (cap-dependent RMSE versus the proposed common-reference
RMSE).  No command in this suite evaluates the test set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "strict_rul_train.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root", type=Path, default=Path(r"E:\NASA_Lab\ress_strict")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"]
    )
    parser.add_argument(
        "--caps", nargs="+", type=int, default=[90, 100, 110, 115, 125]
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["own_rmse", "reference_rmse"],
        choices=["own_rmse", "reference_rmse"],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def valid_result(path: Path, expected):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(result.get(key) == value for key, value in expected.items())


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_started = time.time()
    completed = skipped = failed = 0

    for dataset in args.datasets:
        for objective in args.objectives:
            objective_tag = "own" if objective == "own_rmse" else "ref125"
            for cap in args.caps:
                tag = f"capsweep_{objective_tag}_cap{cap}_s{args.split_seed}"
                result_path = (
                    args.output_root / "results" / f"{dataset}_{tag}.json"
                )
                expected = {
                    "protocol": "strict-rul-cap-v1",
                    "dataset": dataset,
                    "tag": tag,
                    "train_cap": float(cap),
                    "reporting_cap": 125.0,
                    "checkpoint_metric": objective,
                    "validation_only": True,
                    "split_seed": args.split_seed,
                }
                if valid_result(result_path, expected):
                    print(f"SKIP {dataset} {objective} cap={cap}", flush=True)
                    skipped += 1
                    continue

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
                    "--tag",
                    tag,
                    "--train_cap",
                    str(cap),
                    "--reporting_cap",
                    "125",
                    "--checkpoint_metric",
                    objective,
                    "--validation_only",
                    "--seeds",
                    str(args.seeds),
                    "--seed_base",
                    str(args.seed_base),
                    "--split_seed",
                    str(args.split_seed),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--disable_het",
                    "--disable_length",
                ]
                log_path = log_dir / f"{dataset}_{tag}.log"
                print(
                    f"RUN  {dataset} {objective} cap={cap} -> {log_path}",
                    flush=True,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=args.data_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if process.returncode == 0 and valid_result(result_path, expected):
                    completed += 1
                else:
                    failed += 1
                    print(
                        f"FAIL {dataset} {objective} cap={cap}; see {log_path}",
                        flush=True,
                    )

    summary = {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - queue_started,
        "datasets": args.datasets,
        "caps": args.caps,
        "objectives": args.objectives,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    summary_path = args.output_root / "strict_cap_sweep_queue_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
