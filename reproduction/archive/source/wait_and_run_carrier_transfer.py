"""Queue the validation-only second-carrier check after TPE robustness."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--campaign_root", type=Path, required=True)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    prerequisite = campaign / "tpe_robustness_summary.json"
    while not prerequisite.is_file():
        print(f"WAIT {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_carrier_transfer.py"),
            "--data_root",
            str(args.data_root.resolve()),
            "--campaign_root",
            str(campaign),
            "--workers",
            str(args.workers),
        ],
        cwd=HERE,
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
