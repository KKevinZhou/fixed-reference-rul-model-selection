"""Verify Proposition 1 and reproduce the archived C-MAPSS diagnostic table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


VALID_CAPS = {90, 100, 110, 115, 125}
DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).resolve().parent / "analysis"
    )
    return parser.parse_args()


def load_rows(results_dir: Path):
    rows = []
    for dataset in DATASETS:
        pattern = f"{dataset}_V_A_gate_cap*.json"
        for path in sorted(results_dir.glob(pattern)):
            match = re.search(r"cap(\d+)", path.stem)
            if not match:
                continue
            cap = int(match.group(1))
            # Processed source labels were capped at 125. Runs above 125 did not
            # represent different training targets and are intentionally excluded.
            if cap not in VALID_CAPS:
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            prediction = np.asarray(result["ensemble_unit_preds"], dtype=np.float64)
            original = np.asarray(result["test_true_uncapped"], dtype=np.float64)
            prediction = np.clip(prediction, 0.0, cap)
            self_target = np.clip(original, 0.0, cap)

            self_mse = float(np.mean((prediction - self_target) ** 2))
            original_mse = float(np.mean((prediction - original) ** 2))
            exposure_sq = float(np.mean(np.maximum(original - cap, 0.0) ** 2))
            hidden_mse = original_mse - self_mse
            tolerance = 1e-8 * max(1.0, original_mse)
            if hidden_mse + tolerance < exposure_sq:
                raise AssertionError(
                    f"Proposition 1 failed for {dataset}, cap={cap}: "
                    f"{hidden_mse} < {exposure_sq}"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "cap": cap,
                    "n_test_units": len(original),
                    "saturated_fraction": float(np.mean(original > cap)),
                    "self_rmse": float(np.sqrt(self_mse)),
                    "original_rmse": float(np.sqrt(original_mse)),
                    "exposure_bound": float(np.sqrt(exposure_sq)),
                    "hidden_mse": hidden_mse,
                    "bound_fraction_of_hidden_mse": (
                        exposure_sq / hidden_mse if hidden_mse > 0 else 1.0
                    ),
                    "source_file": path.name,
                }
            )
    return sorted(rows, key=lambda row: (row["dataset"], row["cap"]))


def main():
    args = parse_args()
    rows = load_rows(args.results_dir)
    if len(rows) != len(DATASETS) * len(VALID_CAPS):
        raise RuntimeError(f"Expected 20 valid rows, found {len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "exposure_bound_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cap90 = [row for row in rows if row["cap"] == 90]
    summary = {
        "proposition_verified_for_all_rows": True,
        "n_rows": len(rows),
        "valid_caps": sorted(VALID_CAPS),
        "cap90": {
            "saturated_fraction_range": [
                min(row["saturated_fraction"] for row in cap90),
                max(row["saturated_fraction"] for row in cap90),
            ],
            "self_rmse_range": [
                min(row["self_rmse"] for row in cap90),
                max(row["self_rmse"] for row in cap90),
            ],
            "original_rmse_range": [
                min(row["original_rmse"] for row in cap90),
                max(row["original_rmse"] for row in cap90),
            ],
            "exposure_bound_range": [
                min(row["exposure_bound"] for row in cap90),
                max(row["exposure_bound"] for row in cap90),
            ],
            "bound_fraction_of_hidden_mse_range": [
                min(row["bound_fraction_of_hidden_mse"] for row in cap90),
                max(row["bound_fraction_of_hidden_mse"] for row in cap90),
            ],
        },
    }
    summary_path = args.output_dir / "exposure_bound_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
