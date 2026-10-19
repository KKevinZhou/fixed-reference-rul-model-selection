"""Wait for screening completion, then run the preregistered seed-stability stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STABILITY = SCRIPT_DIR / "run_confirmatory_seed_stability.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()
    summary = args.output_root / "cap_sweep_summary.json"
    while not summary.exists():
        print(f"WAIT {summary}", flush=True)
        time.sleep(args.poll_seconds)
    command = [
        sys.executable,
        "-u",
        str(STABILITY),
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
    ]
    process = subprocess.run(command, cwd=SCRIPT_DIR)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
