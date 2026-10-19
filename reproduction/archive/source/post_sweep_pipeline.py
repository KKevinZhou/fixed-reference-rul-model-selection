"""Continue the strict experiment campaign after the cap sweep finishes.

The script waits for a successful cap-sweep summary, locks the best cap under
each validation objective, performs one five-seed test evaluation per locked
configuration, and then runs validation-only TPE plus its locked final test on
FD001 and FD002.
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
HPO = SCRIPT_DIR / "strict_hpo.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root", type=Path, default=Path(r"E:\NASA_Lab\ress_strict")
    )
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--final_seeds", type=int, default=5)
    parser.add_argument("--hpo_trials", type=int, default=20)
    return parser.parse_args()


def run_logged(command, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def wait_for_sweep(args):
    summary_path = args.output_root / "strict_cap_sweep_queue_summary.json"
    while not summary_path.exists():
        print(f"WAIT {summary_path}", flush=True)
        time.sleep(args.poll_seconds)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("failed") != 0:
        raise RuntimeError(f"Cap sweep contains failures: {summary}")
    expected = (
        len(summary["datasets"])
        * len(summary["caps"])
        * len(summary["objectives"])
    )
    if summary.get("completed", 0) + summary.get("skipped", 0) != expected:
        raise RuntimeError(f"Incomplete cap sweep: {summary}")
    return summary


def lock_caps(args, summary):
    locked = {}
    for dataset in summary["datasets"]:
        locked[dataset] = {}
        for objective in summary["objectives"]:
            short = "own" if objective == "own_rmse" else "ref125"
            candidates = []
            for cap in summary["caps"]:
                tag = f"capsweep_{short}_cap{cap}_s{summary['split_seed']}"
                path = args.output_root / "results" / f"{dataset}_{tag}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                value = result["validation_summary"][objective]["mean"]
                candidates.append((value, cap, result))
            value, cap, result = min(candidates, key=lambda item: item[0])
            locked[dataset][objective] = {
                "cap": cap,
                "validation_value": value,
                "source_result": str(
                    args.output_root
                    / "results"
                    / f"{dataset}_{result['tag']}.json"
                ),
            }
    path = args.output_root / "locked_cap_decisions.json"
    path.write_text(json.dumps(locked, indent=2), encoding="utf-8")
    return locked


def final_cap_tests(args, summary, locked):
    for dataset, objectives in locked.items():
        for objective, decision in objectives.items():
            short = "own" if objective == "own_rmse" else "ref125"
            cap = decision["cap"]
            tag = f"locked_{short}_cap{cap}_final5"
            result_path = args.output_root / "results" / f"{dataset}_{tag}.json"
            if result_path.exists():
                print(f"SKIP final {dataset} {objective}", flush=True)
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
                "--seeds",
                str(args.final_seeds),
                "--seed_base",
                "42",
                "--split_seed",
                str(summary["split_seed"]),
                "--epochs",
                "120",
                "--patience",
                "20",
                "--disable_het",
                "--disable_length",
            ]
            log = args.output_root / "final_logs" / f"{dataset}_{tag}.log"
            print(f"RUN final {dataset} {objective} cap={cap}", flush=True)
            run_logged(command, args.data_root, log)


def hpo_suite(args):
    for dataset in ["FD001", "FD002"]:
        for objective in ["own_rmse", "reference_rmse"]:
            common = [
                sys.executable,
                "-u",
                str(HPO),
                "--dataset",
                dataset,
                "--objective",
                objective,
                "--trials",
                str(args.hpo_trials),
                "--trial_seeds",
                "1",
                "--final_seeds",
                str(args.final_seeds),
                "--data_root",
                str(args.data_root),
                "--output_root",
                str(args.output_root),
                "--epochs",
                "100",
                "--patience",
                "15",
            ]
            short = "own" if objective == "own_rmse" else "ref125"
            validation_log = (
                args.output_root / "hpo_suite_logs" / f"{dataset}_{short}_search.log"
            )
            print(f"RUN HPO search {dataset} {objective}", flush=True)
            run_logged(common, args.data_root, validation_log)

            final_log = (
                args.output_root / "hpo_suite_logs" / f"{dataset}_{short}_final.log"
            )
            print(f"RUN HPO locked test {dataset} {objective}", flush=True)
            run_logged(common + ["--final_test"], args.data_root, final_log)


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    summary = wait_for_sweep(args)
    locked = lock_caps(args, summary)
    print(json.dumps(locked, indent=2), flush=True)
    final_cap_tests(args, summary, locked)
    hpo_suite(args)
    completion = {
        "status": "complete",
        "locked_caps": locked,
        "hpo_datasets": ["FD001", "FD002"],
        "hpo_trials": args.hpo_trials,
        "completed_at_unix": time.time(),
    }
    path = args.output_root / "post_sweep_pipeline_summary.json"
    path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
