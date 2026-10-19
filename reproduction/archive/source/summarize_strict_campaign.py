"""Collect strict cap-sweep and locked-test JSON files into tidy tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CAP_PATTERN = re.compile(
    r"(?P<dataset>FD00[1-4])_capsweep_(?P<short>own|ref125)_"
    r"cap(?P<cap>\d+)_s(?P<split>\d+)\.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(r"E:\NASA_Lab\ress_strict")
    )
    return parser.parse_args()


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_dir = args.root / "results"
    summary_dir = args.root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(results_dir.glob("FD00*_capsweep_*.json")):
        match = CAP_PATTERN.fullmatch(path.name)
        if not match:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        validation = result["validation_summary"]
        rows.append(
            {
                "dataset": match.group("dataset"),
                "objective": result["checkpoint_metric"],
                "cap": int(match.group("cap")),
                "split_seed": int(match.group("split")),
                "n_model_seeds": len(result["model_seeds"]),
                "own_rmse_mean": validation["own_rmse"]["mean"],
                "own_rmse_std": validation["own_rmse"]["std"],
                "reference_rmse_mean": validation["reference_rmse"]["mean"],
                "reference_rmse_std": validation["reference_rmse"]["std"],
                "original_rmse_mean": validation["original_rmse"]["mean"],
                "original_rmse_std": validation["original_rmse"]["std"],
                "saturation_fraction": validation["saturation_fraction"]["mean"],
                "exposure_bound": validation["exposure_bound"]["mean"],
                "source_file": path.name,
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["objective"], row["cap"]))
    write_csv(summary_dir / "strict_cap_sweep.csv", rows)

    locked = []
    groups = sorted({(row["dataset"], row["objective"]) for row in rows})
    for dataset, objective in groups:
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["objective"] == objective
        ]
        metric = (
            "own_rmse_mean"
            if objective == "own_rmse"
            else "reference_rmse_mean"
        )
        best = min(subset, key=lambda row: row[metric])
        locked.append(
            {
                "dataset": dataset,
                "objective": objective,
                "selected_cap": best["cap"],
                "validation_value": best[metric],
                "selection_metric": metric,
                "n_candidates_available": len(subset),
            }
        )
    write_csv(summary_dir / "locked_cap_decisions.csv", locked)

    final_rows = []
    for path in sorted(results_dir.glob("FD00*_locked_*_final5.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        ensemble = result["ensemble_test"]
        test = result["test_summary"]
        final_rows.append(
            {
                "dataset": result["dataset"],
                "objective": result["checkpoint_metric"],
                "cap": result["train_cap"],
                "n_model_seeds": len(result["model_seeds"]),
                "test_own_rmse_mean": test["own_rmse"]["mean"],
                "test_own_rmse_std": test["own_rmse"]["std"],
                "test_reference_rmse_mean": test["reference_rmse"]["mean"],
                "test_reference_rmse_std": test["reference_rmse"]["std"],
                "test_original_rmse_mean": test["original_rmse"]["mean"],
                "test_original_rmse_std": test["original_rmse"]["std"],
                "ensemble_own_rmse": ensemble["own_rmse"],
                "ensemble_reference_rmse": ensemble["reference_rmse"],
                "ensemble_original_rmse": ensemble["original_rmse"],
                "saturation_fraction": ensemble["saturation_fraction"],
                "exposure_bound": ensemble["exposure_bound"],
                "source_file": path.name,
            }
        )
    write_csv(summary_dir / "locked_cap_test_results.csv", final_rows)

    status = {
        "cap_sweep_rows": len(rows),
        "cap_decisions": locked,
        "locked_test_rows": len(final_rows),
    }
    (summary_dir / "campaign_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
