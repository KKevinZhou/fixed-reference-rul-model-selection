"""Resume the immutable cap screen with independent cells in parallel.

The wrapper waits for explicitly named trainer PIDs, validates the existing
campaign manifest, fills only missing dataset/cap/split cells with the frozen
canonical runner, and finally invokes the canonical driver to revalidate all
60 cells and create the decision lock.  It changes scheduling only.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_confirmatory_cap_sweep as sweep


SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_DRIVER = SCRIPT_DIR / "run_confirmatory_cap_sweep.py"


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_pids(pids: list[int], poll_seconds: int) -> None:
    while True:
        alive = [pid for pid in pids if process_alive(pid)]
        if not alive:
            return
        print(f"WAIT trainer_pids={alive}", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--wait_pids", type=int, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--validation_interval", type=int, default=2)
    parser.add_argument("--carrier", default="compact_tcn")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise RuntimeError("Cap-screen concurrency is restricted to 1--3 workers")
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    wait_for_pids(args.wait_pids, args.poll_seconds)

    manifest_path = args.output_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Existing campaign manifest is missing")
    expected = sweep.build_manifest(args)
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != expected:
        raise RuntimeError("Existing campaign manifest differs from arguments")

    cells = [
        (dataset, cap, split_seed)
        for dataset in sweep.DATASETS
        for split_seed in sweep.SPLIT_SEEDS
        for cap in sweep.CAPS
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                sweep.run_one,
                args,
                existing,
                dataset,
                cap,
                split_seed,
            ): (dataset, cap, split_seed)
            for dataset, cap, split_seed in cells
        }
        for future in as_completed(futures):
            dataset, cap, split_seed = futures[future]
            path = future.result()
            print(
                f"DONE {dataset} cap={cap} split={split_seed} path={path}",
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
    ]
    process = subprocess.run(command, cwd=SCRIPT_DIR)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
