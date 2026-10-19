"""Wait for screening, prefill independent stability runs in parallel, then lock.

This wrapper does not change any candidate, seed, split, budget, trainer, or
selection rule.  It calls the frozen seed-stability implementation for each
independent dataset/cap cell and finally reruns the canonical driver, which
validates every cached result before creating the decision lock.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_confirmatory_seed_stability as stability


SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_DRIVER = SCRIPT_DIR / "run_confirmatory_seed_stability.py"


def wait_for(path: Path, poll_seconds: int) -> None:
    while not path.exists():
        print(f"WAIT {path}", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise RuntimeError("Stability concurrency is restricted to 1--3 workers")
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()

    wait_for(args.output_root / "cap_sweep_summary.json", args.poll_seconds)
    screening_path = args.output_root / "campaign_manifest.json"
    screening_lock_path = args.output_root / "locked_cap_decisions.json"
    if not screening_path.is_file() or not screening_lock_path.is_file():
        raise RuntimeError("Three-split screening campaign is not locked")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if screening.get("trainer_sha256") != stability.sha256_file(
        stability.TRAINER
    ):
        raise RuntimeError("Trainer changed after the screening campaign")

    manifest = stability.stability_manifest(screening)
    manifest_path = args.output_root / "stability_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing stability manifest differs")
    else:
        stability.exclusive_json(manifest_path, manifest)

    cells = [
        (dataset, cap)
        for dataset in stability.DATASETS
        for cap in stability.CAPS
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                stability.run_additional,
                args,
                screening,
                manifest,
                dataset,
                cap,
            ): (dataset, cap)
            for dataset, cap in cells
        }
        for future in as_completed(futures):
            dataset, cap = futures[future]
            path = future.result()
            print(
                f"DONE stability {dataset} cap={cap} path={path}",
                flush=True,
            )

    command = [
        sys.executable,
        "-u",
        str(CANONICAL_DRIVER),
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
    ]
    process = subprocess.run(command, cwd=SCRIPT_DIR)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
