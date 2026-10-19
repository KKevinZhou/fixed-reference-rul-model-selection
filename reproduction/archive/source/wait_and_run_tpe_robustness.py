"""Wait for TPE lock completion, then run frozen-config robustness checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBUSTNESS = SCRIPT_DIR / "run_tpe_robustness.py"


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
    marker = args.output_root / "TPE_SUITE_COMPLETE"
    while not marker.exists():
        print(f"WAIT {marker}", flush=True)
        time.sleep(args.poll_seconds)
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROBUSTNESS),
            "--data_root",
            str(args.data_root),
            "--output_root",
            str(args.output_root),
        ],
        cwd=SCRIPT_DIR,
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
