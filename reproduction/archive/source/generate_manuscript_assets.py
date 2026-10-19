"""Generate confirmatory manuscript tables and figures from locked CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from run_confirmatory_final import validate_completed_chain


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
METHODS = [
    "cap_grid_self_scaled",
    "cap_grid_fixed_reference",
    "fixed_cap_125",
]
LABELS = {
    "self_scaled": "Self-scaled",
    "fixed_reference": "Fixed-reference",
    "fixed_cap_125": "Fixed $c=125$",
    "cap_grid_self_scaled": "Self-scaled",
    "cap_grid_fixed_reference": "Fixed-reference",
    "tpe_self_scaled": "TPE self-scaled",
    "tpe_fixed_reference": "TPE fixed-reference",
    "tpe_fixed_cap_125": "TPE fixed $c=125$",
}
COLORS = {
    "cap_grid_self_scaled": "#0072B2",
    "cap_grid_fixed_reference": "#D55E00",
    "fixed_cap_125": "#666666",
}
HATCHES = {
    "cap_grid_self_scaled": "",
    "cap_grid_fixed_reference": "//",
    "fixed_cap_125": "..",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing locked summary table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Locked summary table is empty: {path}")
    return frame


def require_unique(
    frame: pd.DataFrame,
    keys: list[str],
    expected_rows: int,
    label: str,
) -> None:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing key columns: {missing}")
    if len(frame) != expected_rows or frame.duplicated(keys).any():
        raise RuntimeError(
            f"{label} must contain {expected_rows} unique {keys} rows"
        )


def require_exact_key_set(
    frame: pd.DataFrame,
    keys: list[str],
    expected: set[tuple],
    label: str,
) -> None:
    observed = set(
        zip(*(frame[key].tolist() for key in keys), strict=True)
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"{label} key matrix is invalid; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def require_finite_columns(
    frame: pd.DataFrame, columns: list[str], label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing numeric columns: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains non-finite numeric values")


def validate_reportable_metrics(finals, pairs, validation) -> None:
    rmse_columns = [
        "own_rmse_ensemble",
        "candidate_target_common_support_rmse_ensemble",
        "candidate_support_reference_rmse_ensemble",
        "reference_rmse_ensemble",
        "reference_rmse_mean",
        "reference_rmse_std",
        "reference_rmse_bootstrap_low",
        "reference_rmse_bootstrap_high",
        "nasa_score_reference_ensemble",
        "saturation_fraction",
        "exposure_bound",
        "reference_exposure_bound",
        "mean_error_ensemble",
        "late_prediction_fraction",
    ]
    for threshold in (20, 30, 50):
        rmse_columns.extend(
            [
                f"n_due_at_{threshold}",
                f"n_not_due_at_{threshold}",
                f"delayed_action_rate_at_{threshold}",
                f"premature_action_rate_at_{threshold}",
            ]
        )
    require_finite_columns(finals, rmse_columns, "Final-test table")
    nonnegative = [
        column
        for column in rmse_columns
        if column not in {"mean_error_ensemble"}
    ]
    if (finals[nonnegative].to_numpy(dtype=float) < 0).any():
        raise RuntimeError("Final-test nonnegative metrics contain negatives")
    fraction_columns = [
        "saturation_fraction",
        "late_prediction_fraction",
        *[
            f"{kind}_action_rate_at_{threshold}"
            for threshold in (20, 30, 50)
            for kind in ("delayed", "premature")
        ],
    ]
    fractions = finals[fraction_columns].to_numpy(dtype=float)
    if ((fractions < 0) | (fractions > 1)).any():
        raise RuntimeError("Final-test fractions must lie in [0, 1]")
    for threshold in (20, 30, 50):
        if (
            finals[
                [f"n_due_at_{threshold}", f"n_not_due_at_{threshold}"]
            ].to_numpy(dtype=float)
            <= 0
        ).any():
            raise RuntimeError("Maintenance-proxy strata must be nonempty")
    low = finals["reference_rmse_bootstrap_low"].to_numpy(dtype=float)
    high = finals["reference_rmse_bootstrap_high"].to_numpy(dtype=float)
    if (low > high).any():
        raise RuntimeError("Single-method bootstrap interval is reversed")
    for column in (
        "bound_fraction_of_hidden_gap",
        "reference_bound_fraction_of_hidden_gap",
    ):
        if column not in finals:
            raise RuntimeError(f"Final-test table is missing {column}")
        values = pd.to_numeric(finals[column], errors="coerce").dropna()
        if ((values < -1e-10) | (values > 1.0 + 1e-8)).any():
            raise RuntimeError(f"{column} lies outside its theoretical range")

    pair_columns = [
        "mean_difference",
        "paired_engine_rmse_difference",
        "paired_engine_bootstrap_95ci_low",
        "paired_engine_bootstrap_95ci_high",
        "bootstrap_fraction_negative",
        "wilcoxon_holm_all12_pvalue",
    ]
    require_finite_columns(pairs, pair_columns, "Paired-test table")
    pair_low = pairs["paired_engine_bootstrap_95ci_low"].to_numpy(dtype=float)
    pair_high = pairs["paired_engine_bootstrap_95ci_high"].to_numpy(dtype=float)
    if (pair_low > pair_high).any():
        raise RuntimeError("Paired bootstrap interval is reversed")
    for column in (
        "bootstrap_fraction_negative",
        "wilcoxon_holm_primary_pvalue",
        "wilcoxon_holm_secondary_pvalue",
        "wilcoxon_holm_all12_pvalue",
    ):
        if column not in pairs:
            raise RuntimeError(f"Paired-test table is missing {column}")
        values = pd.to_numeric(pairs[column], errors="coerce").dropna()
        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"{column} must lie in [0, 1]")

    require_finite_columns(
        validation,
        [
            "saturation_fraction",
            "reference_exposure_bound",
            "exposure_bound",
        ],
        "Validation diagnostics",
    )
    if (
        (validation["saturation_fraction"] < 0)
        | (validation["saturation_fraction"] > 1)
        | (validation["reference_exposure_bound"] < 0)
        | (validation["exposure_bound"] < 0)
    ).any():
        raise RuntimeError("Validation diagnostics are outside valid ranges")


def validate_summary_provenance(
    summary: Path, chain_validator=validate_completed_chain
) -> dict:
    status_path = summary / "summary_status.json"
    if not status_path.is_file():
        raise RuntimeError("Locked summary provenance is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_counts = {
        "validation_rows": 120,
        "stability_rows": 80,
        "bound_rows": 120,
        "cap_rows": 40,
        "aggregation_sensitivity_rows": 32,
        "decision_rows": 12,
        "tpe_rows": 6,
        "tpe_robustness_rows": 6,
        "decomposition_rows": 160,
        "final_rows": 18,
        "paired_tests": 12,
    }
    if (
        status.get("protocol") != "confirmatory-summary-v2"
        or status.get("complete") is not True
        or status.get("bound_failures") != 0
        or any(status.get(key) != value for key, value in expected_counts.items())
        or not status.get("campaign_id")
        or not status.get("final_test_summary_sha256")
        or not status.get("test_evaluated_marker_sha256")
        or status.get("summarizer_sha256")
        != sha256_file(Path(__file__).resolve().with_name(
            "summarize_confirmatory.py"
        ))
        or status.get("manuscript_asset_generator_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError(f"Locked summary status is incomplete: {status}")
    campaign_root = summary.parent
    final_summary = campaign_root / "final_test_summary.json"
    test_marker = campaign_root / "TEST_EVALUATED"
    plan_path = campaign_root / "immutable_test_plan.json"
    started_path = campaign_root / "TEST_EVALUATION_STARTED.json"
    if (
        not final_summary.is_file()
        or sha256_file(final_summary)
        != status["final_test_summary_sha256"]
        or not test_marker.is_file()
        or sha256_file(test_marker)
        != status["test_evaluated_marker_sha256"]
        or not plan_path.is_file()
        or not started_path.is_file()
    ):
        raise RuntimeError(
            "Summary provenance no longer matches final campaign artefacts"
        )
    chain_validator(
        json.loads(final_summary.read_text(encoding="utf-8")),
        json.loads(test_marker.read_text(encoding="utf-8")),
        json.loads(plan_path.read_text(encoding="utf-8")),
        campaign_root,
        started_path,
    )
    expected_hashes = status.get("csv_sha256", {})
    required_csv = {
        "validation_long.csv",
        "stability_long.csv",
        "exposure_bound_audit.csv",
        "cap_summary.csv",
        "aggregation_sensitivity.csv",
        "locked_decisions.csv",
        "tpe_locks.csv",
        "tpe_robustness.csv",
        "target_support_decomposition.csv",
        "final_test.csv",
        "paired_tests.csv",
    }
    if set(expected_hashes) != required_csv:
        raise RuntimeError(
            "Summary status CSV set is incomplete or contains unplanned files"
        )
    for name in required_csv:
        path = summary / name
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"Summary CSV hash mismatch: {path}")
    return status


def validation_figure(cap_summary, destination):
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        subset = cap_summary[cap_summary["dataset"] == dataset]
        for objective, color, marker in (
            ("self_scaled", "#0072B2", "o"),
            ("fixed_reference", "#D55E00", "s"),
        ):
            rows = subset[subset["objective"] == objective].sort_values(
                "cap"
            )
            if len(rows) != 5:
                raise RuntimeError(
                    f"Incomplete cap profile for {dataset} {objective}"
                )
            axis.errorbar(
                rows["cap"],
                rows["mean_across_splits"],
                yerr=rows["std_across_splits"],
                color=color,
                marker=marker,
                linewidth=1.5,
                capsize=2.5,
                label=LABELS[objective],
            )
        axis.axvline(125, color="#888888", linestyle=":", linewidth=1)
        axis.set_title(dataset)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Training cap $c$ (cycles)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Validation RMSE (cycles)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    temporary = destination.with_name(
        f"{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    figure.savefig(temporary, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, destination)


def final_rmse_figure(finals, destination):
    subset = finals[finals["method"].isin(METHODS)].copy()
    if len(subset) != len(DATASETS) * len(METHODS):
        raise RuntimeError("Incomplete locked cap-grid test matrix")
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(DATASETS))
    width = 0.24
    for offset, method in enumerate(METHODS):
        rows = (
            subset[subset["method"] == method]
            .set_index("dataset")
            .loc[DATASETS]
        )
        axis.bar(
            x + (offset - 1) * width,
            rows["reference_rmse_mean"],
            width,
            yerr=rows["reference_rmse_std"],
            color=COLORS[method],
            edgecolor="#222222",
            linewidth=0.5,
            hatch=HATCHES[method],
            capsize=2.5,
            label=LABELS[method],
        )
    axis.set_xticks(x, DATASETS)
    axis.set_ylabel(
        "Plan-bound descriptive reference RMSE on standard test split (cycles)"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    temporary = destination.with_name(
        f"{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    figure.savefig(temporary, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, destination)


def f2(value):
    return f"{float(value):.2f}"


def f3(value):
    return f"{float(value):.3f}"


def f3_optional(value):
    if value is None or pd.isna(value):
        return "--"
    return f3(value)


def final_methods_for_dataset(dataset):
    methods = list(METHODS)
    if dataset in {"FD001", "FD002"}:
        methods.extend(
            [
                "tpe_self_scaled",
                "tpe_fixed_reference",
                "tpe_fixed_cap_125",
            ]
        )
    return methods


def two_by_two_mse_contrasts(row):
    """Return the four cells and all prespecified 2x2 contrasts on MSE scale."""
    cells = {
        "Lcc": float(row["own_rmse_ensemble"]) ** 2,
        "Lcr": float(
            row["candidate_target_common_support_rmse_ensemble"]
        ) ** 2,
        "Lrc": float(
            row["candidate_support_reference_rmse_ensemble"]
        ) ** 2,
        "Lrr": float(row["reference_rmse_ensemble"]) ** 2,
    }
    contrasts = {
        "delta_t_c": cells["Lrc"] - cells["Lcc"],
        "delta_t_r": cells["Lrr"] - cells["Lcr"],
        "delta_s_c": cells["Lcr"] - cells["Lcc"],
        "delta_s_r": cells["Lrr"] - cells["Lrc"],
    }
    interaction_target = (
        contrasts["delta_t_r"] - contrasts["delta_t_c"]
    )
    interaction_support = (
        contrasts["delta_s_r"] - contrasts["delta_s_c"]
    )
    interaction_cells = (
        cells["Lrr"] - cells["Lcr"] - cells["Lrc"] + cells["Lcc"]
    )
    if not np.allclose(
        [interaction_target, interaction_support],
        interaction_cells,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Inconsistent 2x2 interaction calculation")
    return {
        **cells,
        **contrasts,
        "delta_int": interaction_cells,
    }


def latex_tables(decisions, tpe, finals, pairs):
    decision_index = decisions.set_index(["dataset", "objective"])
    final_index = finals.set_index(["dataset", "method"])
    primary_pairs = pairs[
        pairs["comparison"] == "fixed_vs_self_cap_grid"
    ].set_index("dataset")
    if (
        len(primary_pairs) != len(DATASETS)
        or set(primary_pairs.index) != set(DATASETS)
    ):
        raise RuntimeError("Incomplete primary paired-comparison family")
    lines = [
        "% Generated by generate_manuscript_assets.py; do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Caps locked before any standard test-set outcome evaluation. Selection values "
        "are the protocol-specified three-split aggregate validation "
        "RMSE under the corresponding objective. The own-target and "
        "fixed-reference columns are selection criteria on different target "
        "scales and must not be compared horizontally.}",
        "\\label{tab:confirmatory-caps}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Subset & $c_{\\rm self}$ & Own-target val. & $c_{\\rm ref}$ & "
        "Fixed-ref. val. \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        self_row = decision_index.loc[(dataset, "self_scaled")]
        fixed_row = decision_index.loc[(dataset, "fixed_reference")]
        lines.append(
            f"{dataset} & {int(self_row['selected_cap'])} & "
            f"{f2(self_row['selection_value'])} & "
            f"{int(fixed_row['selected_cap'])} & "
            f"{f2(fixed_row['selection_value'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3.5pt}",
            "\\caption{Locked five-seed plan-bound descriptive $2\\times2$ "
            "rescoring diagnostic. "
            "Each fitted model is rescored under the $2\\times2$ combination "
            "of candidate/reference targets ($T_c,T_r$) and "
            "candidate/common prediction-clipping ceilings ($S_c,S_r$). Values are "
            "median-ensemble RMSE in cycles; $\\Delta_{\\rm int}$ is the "
            "target-by-output-ceiling interaction on the MSE scale in cycles$^2$. "
            "The cells and interaction are descriptive rescoring contrasts, "
            "not causal or additive performance attributions.}",
            "\\label{tab:confirmatory-decomposition}",
            "\\begin{tabular}{@{}llrrrrrrr@{}}",
            "\\toprule",
            "Subset & Selection & $c$ & $T_c/S_c$ & $T_c/S_r$ & "
            "$T_r/S_c$ & $T_r/S_r$ & $\\Delta_{\\rm int}$ & Ref.-scale NASA \\\\",
            "\\midrule",
        ]
    )
    for dataset in DATASETS:
        for method in METHODS:
            row = final_index.loc[(dataset, method)]
            contrasts = two_by_two_mse_contrasts(row)
            lines.append(
                f"{dataset} & {LABELS[method]} & {int(row['train_cap'])} & "
                f"{f2(row['own_rmse_ensemble'])} & "
                f"{f2(row['candidate_target_common_support_rmse_ensemble'])} & "
                f"{f2(row['candidate_support_reference_rmse_ensemble'])} & "
                f"{f2(row['reference_rmse_ensemble'])} & "
                f"{f2(contrasts['delta_int'])} & "
                f"{f2(row['nasa_score_reference_ensemble'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Primary fixed-reference versus self-scaled cap-grid "
            "contrasts under the common reporting clip. Differences are fixed-reference "
            "minus self-scaled RMSE, so negative values favour fixed-reference "
            "selection. The engine bootstrap is a 10,000-resample percentile "
            "interval conditional on the fitted five-model ensembles; ``fraction "
            "$<0$'' is the descriptive fraction of those resamples with a "
            "negative difference. Seed-level Wilcoxon and Holm summaries "
            "are reported in the Supplementary Material.}",
            "\\label{tab:primary-paired}",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Subset & Seed-mean $\\Delta$ & Ensemble $\\Delta$ & 95\\% CI & "
            "Bootstrap frac. $\\Delta<0$ \\\\",
            "\\midrule",
        ]
    )
    for dataset in DATASETS:
        row = primary_pairs.loc[dataset]
        lines.append(
            f"{dataset} & {f2(row['mean_difference'])} & "
            f"{f2(row['paired_engine_rmse_difference'])} & "
            f"[{f2(row['paired_engine_bootstrap_95ci_low'])}, "
            f"{f2(row['paired_engine_bootstrap_95ci_high'])}] & "
            f"{f3(row['bootstrap_fraction_negative'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Matched 12-trial TPE studies and locked standard-test "
            "reference RMSE. Both validation scales are shown for every "
            "winner; the objective column identifies which one selected it. "
            "TPE selection uses one model seed; test RMSE is the "
            "mean$\\pm$sample standard deviation over five seeds.}",
            "\\label{tab:tpe-confirmatory}",
            "\\begin{tabular}{@{}llrrrrrrr@{}}",
            "\\toprule",
            "Subset & Objective & $c$ & LR & Batch & $\\delta$ & "
            "Own val. & Ref. val. & Test RMSE \\\\",
            "\\midrule",
        ]
    )
    tpe_index = tpe.set_index(["dataset", "objective"])
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            row = tpe_index.loc[(dataset, objective)]
            final = final_index.loc[(dataset, f"tpe_{objective}")]
            lines.append(
                f"{dataset} & {LABELS[objective]} & "
                f"{int(row['best_train_cap'])} & "
                f"{float(row['best_lr']):.2e} & "
                f"{int(row['best_batch'])} & "
                f"{f1(row['best_huber_delta'])} & "
                f"{f2(row['best_own_rmse'])} & "
                f"{f2(row['best_reference_rmse'])} & "
                f"{f2(final['reference_rmse_mean'])}$\\pm$"
                f"{f2(final['reference_rmse_std'])} \\\\"
            )
        if dataset == "FD001":
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Decision-oriented descriptive test-set rates at the "
            "protocol-specified 30-cycle proxy threshold. Rates are conditional on the "
            "relevant true-RUL stratum and use the five-model median "
            "ensemble.}",
            "\\label{tab:maintenance-risk}",
            "\\begin{tabular}{@{}llrrrr@{}}",
            "\\toprule",
            "Subset & Selection & Mean error & Optimistic fraction & Delayed rate & "
            "Premature rate \\\\",
            "\\midrule",
        ]
    )
    for dataset in DATASETS:
        for method in METHODS:
            row = final_index.loc[(dataset, method)]
            lines.append(
                f"{dataset} & {LABELS[method]} & "
                f"{f2(row['mean_error_ensemble'])} & "
                f"{f3(row['late_prediction_fraction'])} & "
                f"{f3(row['delayed_action_rate_at_30'])} & "
                f"{f3(row['premature_action_rate_at_30'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    if pairs.empty:
        raise RuntimeError("Paired confirmatory comparisons are missing")
    return "\n".join(lines) + "\n"


def supplementary_pair_table(pairs):
    labels = {
        "fixed_vs_self_cap_grid": "Grid: ref.--self",
        "fixed_cap_grid_vs_fixed125": "Grid: ref.--fixed-125",
        "fixed_vs_self_tpe": "TPE: ref.--self",
        "fixed_reference_tpe_vs_fixed_cap_125_tpe": (
            "TPE: ref.--fixed-125"
        ),
    }
    lines = [
        "% Generated by generate_manuscript_assets.py; do not edit.",
        "\\section{Complete protocol-specified paired contrasts}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{All protocol-specified common-reporting-clip paired contrasts. "
        "Differences are left method minus right method. Holm$_{\\rm fam}$ "
        "uses the protocol-specified four-contrast primary family or "
        "eight-contrast secondary family, as applicable; Holm$_{12}$ uses "
        "one adjustment across all 12 contrasts. These are algorithmic "
        "repeatability diagnostics with five paired model seeds, not "
        "inferential evidence or decision criteria.}",
        "\\label{sup:tab:all-paired}",
        "\\scriptsize",
        "\\begin{tabular}{@{}llrlrrr@{}}",
        "\\toprule",
        "Subset & Contrast & Seed $\\Delta$ & Ensemble $\\Delta$ [95\\% CI] & "
        "Bootstrap fraction $\\Delta<0$ & Holm$_{\\rm fam}$ $p$ & "
        "Holm$_{12}$ $p$ \\\\",
        "\\midrule",
    ]
    ordered = pairs.sort_values(["dataset", "comparison"])
    for _, row in ordered.iterrows():
        family_column = (
            "wilcoxon_holm_primary_pvalue"
            if row["comparison"] == "fixed_vs_self_cap_grid"
            else "wilcoxon_holm_secondary_pvalue"
        )
        lines.append(
            f"{row['dataset']} & {labels[row['comparison']]} & "
            f"{f2(row['mean_difference'])} & "
            f"{f2(row['paired_engine_rmse_difference'])} "
            f"[{f2(row['paired_engine_bootstrap_95ci_low'])}, "
            f"{f2(row['paired_engine_bootstrap_95ci_high'])}] & "
            f"{f3(row['bootstrap_fraction_negative'])} & "
            f"{f3_optional(row[family_column])} & "
            f"{f3(row['wilcoxon_holm_all12_pvalue'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Complete seed-level paired RMSE differences for the "
            "contrasts above. Differences follow model seeds 45--49 in "
            "order. Negative values favour the left method. The mean, sample "
            "standard deviation, median, and negative-direction count are "
            "descriptive repeatability summaries.}",
            "\\label{sup:tab:seed-paired}",
            "\\begin{tabular}{@{}llp{4.6cm}rrr@{}}",
            "\\toprule",
            "Subset & Contrast & Five paired differences & Mean$\\pm$SD & "
            "Median & $n_{<0}$/5 \\\\",
            "\\midrule",
        ]
    )
    for _, row in ordered.iterrows():
        differences = np.asarray(
            json.loads(row["paired_difference_values"]), dtype=float
        )
        difference_text = ", ".join(f2(value) for value in differences)
        lines.append(
            f"{row['dataset']} & {labels[row['comparison']]} & "
            f"{difference_text} & {f2(np.mean(differences))}$\\pm$"
            f"{f2(np.std(differences, ddof=1))} & "
            f"{f2(np.median(differences))} & "
            f"{int(np.sum(differences < 0))}/5 \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_individual_table(finals):
    index = finals.set_index(["dataset", "method"])
    lines = [
        "",
        "\\section{Individual standard-test uncertainty summaries}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Reference-scale standard-test results for every locked "
        "method. Seed RMSE is mean$\\pm$sample standard deviation over five "
        "models. The ensemble column reports the five-model median-ensemble "
        "RMSE and its two-sided 95\\% percentile engine-bootstrap interval "
        "(2,000 resamples). Intervals are descriptive conditional summaries.}",
        "\\label{sup:tab:individual-test}",
        "\\begin{tabular}{@{}lllrrr@{}}",
        "\\toprule",
        "Subset & Selection & $c$ & Seed RMSE & Ensemble RMSE [95\\% CI] & "
        "NASA score \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for method in final_methods_for_dataset(dataset):
            row = index.loc[(dataset, method)]
            lines.append(
                f"{dataset} & {LABELS[method]} & {int(row['train_cap'])} & "
                f"{f2(row['reference_rmse_mean'])}$\\pm$"
                f"{f2(row['reference_rmse_std'])} & "
                f"{f2(row['reference_rmse_ensemble'])} "
                f"[{f2(row['reference_rmse_bootstrap_low'])}, "
                f"{f2(row['reference_rmse_bootstrap_high'])}] & "
                f"{f2(row['nasa_score_reference_ensemble'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_validation_two_by_two_table(decomposition):
    index = decomposition.set_index(
        [
            "dataset",
            "checkpoint_objective",
            "cap",
            "target_scale",
            "output_support",
        ]
    )
    lines = [
        "",
        "\\section{Validation-only target--output-ceiling contrasts}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Complete validation-only $2\\times2$ checkpoint rescore "
        "for the 40 dataset--checkpoint--cap configurations. Every entry is "
        "on the MSE scale in cycles$^2$ and is the mean of the three "
        "protocol-specified split-level aggregates. Contrasts follow the "
        "target--output-ceiling definitions in the main article and are descriptive, not "
        "causal.}",
        "\\label{sup:tab:validation-two-by-two}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llrrrrrrrrrr@{}}",
        "\\toprule",
        "Subset & Checkpoint & $c$ & $L_{c,c}$ & $L_{c,r}$ & $L_{r,c}$ & "
        "$L_{r,r}$ & $\\Delta_T^{(c)}$ & $\\Delta_T^{(r)}$ & "
        "$\\Delta_S^{(c)}$ & $\\Delta_S^{(r)}$ & "
        "$\\Delta_{\\rm int}$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            for cap in (90, 100, 110, 115, 125):
                split_cells = {}
                for target, support, name in (
                    ("candidate", "candidate", "Lcc"),
                    ("candidate", "reference", "Lcr"),
                    ("reference", "candidate", "Lrc"),
                    ("reference", "reference", "Lrr"),
                ):
                    try:
                        row = index.loc[
                            (dataset, objective, cap, target, support)
                        ]
                    except KeyError as exc:
                        raise RuntimeError(
                            "Missing validation 2x2 cell for "
                            f"{dataset} {objective} cap={cap} "
                            f"{target}/{support}"
                        ) from exc
                    values = np.asarray(
                        json.loads(row["split_values"]), dtype=float
                    )
                    if values.shape != (3,) or not np.all(np.isfinite(values)):
                        raise RuntimeError(
                            "Validation 2x2 split values must contain three "
                            f"finite aggregates for {dataset} {objective} "
                            f"cap={cap}"
                        )
                    split_cells[name] = values
                contrasts = {
                    "delta_t_c": split_cells["Lrc"] - split_cells["Lcc"],
                    "delta_t_r": split_cells["Lrr"] - split_cells["Lcr"],
                    "delta_s_c": split_cells["Lcr"] - split_cells["Lcc"],
                    "delta_s_r": split_cells["Lrr"] - split_cells["Lrc"],
                }
                interaction = (
                    split_cells["Lrr"]
                    - split_cells["Lcr"]
                    - split_cells["Lrc"]
                    + split_cells["Lcc"]
                )
                if not np.allclose(
                    contrasts["delta_t_r"] - contrasts["delta_t_c"],
                    interaction,
                    rtol=1e-12,
                    atol=1e-12,
                ) or not np.allclose(
                    contrasts["delta_s_r"] - contrasts["delta_s_c"],
                    interaction,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise RuntimeError("Validation 2x2 interaction mismatch")
                lines.append(
                    f"{dataset} & {LABELS[objective]} & {cap} & "
                    f"{f2(np.mean(split_cells['Lcc']))} & "
                    f"{f2(np.mean(split_cells['Lcr']))} & "
                    f"{f2(np.mean(split_cells['Lrc']))} & "
                    f"{f2(np.mean(split_cells['Lrr']))} & "
                    f"{f2(np.mean(contrasts['delta_t_c']))} & "
                    f"{f2(np.mean(contrasts['delta_t_r']))} & "
                    f"{f2(np.mean(contrasts['delta_s_c']))} & "
                    f"{f2(np.mean(contrasts['delta_s_r']))} & "
                    f"{f2(np.mean(interaction))} \\\\"
                )
            if objective == "self_scaled":
                lines.append("\\addlinespace")
        if dataset != DATASETS[-1]:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def checkpoint_pool_selection_rows(decomposition):
    """Rerank caps within each archived checkpoint pool under all four cells."""
    output = []
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            for target, support in (
                ("candidate", "candidate"),
                ("candidate", "reference"),
                ("reference", "candidate"),
                ("reference", "reference"),
            ):
                subset = decomposition[
                    (decomposition["dataset"] == dataset)
                    & (
                        decomposition["checkpoint_objective"]
                        == objective
                    )
                    & (decomposition["target_scale"] == target)
                    & (decomposition["output_support"] == support)
                ].copy()
                if (
                    len(subset) != 5
                    or set(subset["cap"].astype(int))
                    != {90, 100, 110, 115, 125}
                    or not np.isfinite(
                        subset["mean_across_splits"].to_numpy(dtype=float)
                    ).all()
                ):
                    raise RuntimeError(
                        "Incomplete checkpoint-pool selection diagnostic for "
                        f"{dataset} {objective} {target}/{support}"
                    )
                subset["cap"] = subset["cap"].astype(int)
                subset["mean_across_splits"] = subset[
                    "mean_across_splits"
                ].astype(float)
                best = subset.sort_values(
                    ["mean_across_splits", "cap"],
                    ascending=[True, False],
                    kind="mergesort",
                ).iloc[0]
                output.append(
                    {
                        "dataset": dataset,
                        "checkpoint_objective": objective,
                        "target_scale": target,
                        "output_support": support,
                        "selected_cap": int(best["cap"]),
                        "selection_mse": float(
                            best["mean_across_splits"]
                        ),
                        "selection_rmse": float(
                            np.sqrt(best["mean_across_splits"])
                        ),
                    }
                )
    return output


def supplementary_checkpoint_pool_selection_table(decomposition):
    """Format the archived-checkpoint-pool cap-ranking sensitivity."""
    selection = {
        (
            row["dataset"],
            row["checkpoint_objective"],
            row["target_scale"],
            row["output_support"],
        ): row
        for row in checkpoint_pool_selection_rows(decomposition)
    }
    lines = [
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Post-hoc cap-ranking sensitivity within each archived "
        "checkpoint pool. Each entry is selected cap (RMSE in cycles), where "
        "RMSE is the square root of the three-split mean MSE. $T_c$ and $T_r$ "
        "denote candidate and reference targets; $S_c$ and $S_r$ denote "
        "candidate and common output supports. This diagnostic reranks the "
        "five caps among checkpoints already selected by the named original "
        "objective; it does not reconstruct unarchived epoch histories or "
        "constitute an independently executed selection arm.}",
        "\\label{sup:tab:checkpoint-pool-selection}",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Subset & Checkpoint pool & $T_c,S_c$ & $T_c,S_r$ & "
        "$T_r,S_c$ & $T_r,S_r$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            entries = []
            for target, support in (
                ("candidate", "candidate"),
                ("candidate", "reference"),
                ("reference", "candidate"),
                ("reference", "reference"),
            ):
                best = selection[(dataset, objective, target, support)]
                entries.append(
                    f"{best['selected_cap']} ({f2(best['selection_rmse'])})"
                )
            lines.append(
                f"{dataset} & {LABELS[objective]} & "
                + " & ".join(entries)
                + " \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_aggregation_sensitivity_table(frame):
    schemes = (
        ("split_equal", "Split-equal"),
        ("run_equal", "Run-equal"),
        ("primary_three_seed_mean", "Primary 3-seed"),
        ("seed42_three_split_mean", "Seed-42 splits"),
    )
    lookup = {
        (
            row.dataset,
            row.objective,
            row.aggregation_scheme,
        ): int(row.selected_cap)
        for row in frame.itertuples(index=False)
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
            "\\caption{Split/seed sensitivity of validation-stage cap selection under four "
        "aggregation schemes. The protocol-specified rule is split-equal; the "
            "other columns are report-only sensitivity checks.}",
        "\\label{sup:tab:aggregation-sensitivity}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Subset & Objective & Split-equal & Run-equal & Primary 3-seed & "
        "Seed-42 splits \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            values = [
                lookup[(dataset, objective, scheme)]
                for scheme, _ in schemes
            ]
            lines.append(
                f"{dataset} & {LABELS[objective]} & "
                + " & ".join(str(value) for value in values)
                + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_tpe_robustness_table(frame):
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Report-only validation scores of the six frozen TPE "
        "winners. Selection used seed 42 on split 20260724. The remaining "
        "columns are post-lock evaluations and did not trigger reselection. "
        "Scores use each study's declared selection metric.}",
        "\\label{sup:tab:tpe-robustness}",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Subset & Study & Selection & Seed 43 & Seed 44 & Split 20260725 & "
        "Split 20260726 \\\\",
        "\\midrule",
    ]
    indexed = frame.set_index(["dataset", "objective"])
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            row = indexed.loc[(dataset, objective)]
            lines.append(
                f"{dataset} & {LABELS[objective]} & "
                f"{f2(row['selection_value'])} & "
                f"{f2(row['primary_seed43_value'])} & "
                f"{f2(row['primary_seed44_value'])} & "
                f"{f2(row['secondary_split20260725_seed42'])} & "
                f"{f2(row['secondary_split20260726_seed42'])} \\\\"
            )
        if dataset == "FD001":
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_validation_saturation_table(validation):
    required = {
        "dataset",
        "split_seed",
        "cap",
        "saturation_fraction",
        "reference_exposure_bound",
        "exposure_bound",
    }
    missing = required - set(validation.columns)
    if missing:
        raise RuntimeError(
            "Validation saturation table is missing columns: "
            f"{sorted(missing)}"
        )
    collapsed = []
    for (dataset, cap, split), group in validation.groupby(
        ["dataset", "cap", "split_seed"], sort=True
    ):
        values = {}
        for column in (
            "saturation_fraction",
            "reference_exposure_bound",
            "exposure_bound",
        ):
            unique = np.unique(group[column].to_numpy(dtype=float))
            if len(unique) != 1:
                raise RuntimeError(
                    "Target-derived validation diagnostics disagree across "
                    f"objectives for {dataset} cap={cap} split={split}"
                )
            values[column] = float(unique[0])
        collapsed.append(
            {
                "dataset": dataset,
                "cap": int(cap),
                "split_seed": int(split),
                **values,
            }
        )
    collapsed = pd.DataFrame(collapsed)
    if (
        len(collapsed) != len(DATASETS) * 5 * 3
        or set(collapsed["dataset"]) != set(DATASETS)
    ):
        raise RuntimeError(
            "Validation saturation diagnostics must cover 4 datasets, "
            "5 caps, and 3 grouped splits"
        )
    lines = [
        "",
        "\\section{Validation-stage saturation diagnostics}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Validation-stage target saturation and exposure across "
        "the three protocol-specified grouped splits. Values are "
        "mean$\\pm$sample standard deviation of the three split-level "
        "endpoint aggregates. $q_{\\rm val}$ is the fraction of validation "
        "endpoints above candidate cap $c$; $E_{r,\\rm val}$ and "
        "$E_{\\rm val}$ are in cycles.}",
        "\\label{sup:tab:validation-saturation}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Subset & $c$ & $q_{\\rm val}$ (\\%) & $E_{r,\\rm val}$ & "
        "$E_{\\rm val}$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for cap in (90, 100, 110, 115, 125):
            group = collapsed[
                (collapsed["dataset"] == dataset)
                & (collapsed["cap"] == cap)
            ]
            if len(group) != 3:
                raise RuntimeError(
                    f"Incomplete validation diagnostics for {dataset} cap={cap}"
                )
            q = 100.0 * group["saturation_fraction"].to_numpy(dtype=float)
            er = group["reference_exposure_bound"].to_numpy(dtype=float)
            e = group["exposure_bound"].to_numpy(dtype=float)
            lines.append(
                f"{dataset} & {cap} & "
                f"{f2(np.mean(q))}$\\pm${f2(np.std(q, ddof=1))} & "
                f"{f2(np.mean(er))}$\\pm${f2(np.std(er, ddof=1))} & "
                f"{f2(np.mean(e))}$\\pm${f2(np.std(e, ddof=1))} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_two_by_two_contrast_table(finals):
    index = finals.set_index(["dataset", "method"])
    lines = [
        "",
        "\\section{Complete MSE-scale target--output-ceiling contrasts}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Complete standard-test $2\\times2$ rescoring "
        "diagnostic for all 18 locked final methods. The four cells and all "
        "contrasts are on the MSE scale in cycles$^2$. "
        "$\\Delta_T^{(c)}=L_{r,c}-L_{c,c}$, "
        "$\\Delta_T^{(r)}=L_{r,r}-L_{c,r}$, "
        "$\\Delta_S^{(c)}=L_{c,r}-L_{c,c}$, "
        "$\\Delta_S^{(r)}=L_{r,r}-L_{r,c}$, and "
        "$\\Delta_{\\rm int}=\\Delta_T^{(r)}-\\Delta_T^{(c)}="
        "\\Delta_S^{(r)}-\\Delta_S^{(c)}$. These are descriptive rescoring "
        "contrasts, not causal effects.}",
        "\\label{sup:tab:two-by-two-contrasts}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llrrrrrrrrrr@{}}",
        "\\toprule",
        "Subset & Selection & $c$ & $L_{c,c}$ & $L_{c,r}$ & $L_{r,c}$ & "
        "$L_{r,r}$ & $\\Delta_T^{(c)}$ & $\\Delta_T^{(r)}$ & "
        "$\\Delta_S^{(c)}$ & $\\Delta_S^{(r)}$ & "
        "$\\Delta_{\\rm int}$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for method in final_methods_for_dataset(dataset):
            row = index.loc[(dataset, method)]
            values = two_by_two_mse_contrasts(row)
            lines.append(
                f"{dataset} & {LABELS[method]} & {int(row['train_cap'])} & "
                f"{f2(values['Lcc'])} & {f2(values['Lcr'])} & "
                f"{f2(values['Lrc'])} & {f2(values['Lrr'])} & "
                f"{f2(values['delta_t_c'])} & "
                f"{f2(values['delta_t_r'])} & "
                f"{f2(values['delta_s_c'])} & "
                f"{f2(values['delta_s_r'])} & "
                f"{f2(values['delta_int'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_maintenance_threshold_table(finals):
    index = finals.set_index(["dataset", "method"])
    lines = [
        "",
        "\\section{Maintenance-proxy threshold sensitivity}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Finite-benchmark denominators for the conditional "
        "maintenance-action proxy rates. The target strata are common to "
        "all locked methods within a subset.}",
        "\\label{sup:tab:maintenance-denominators}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Subset & Threshold $h$ & $n(T_r\\leq h)$ & $n(T_r>h)$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        rows = finals[finals["dataset"] == dataset]
        for threshold in (20, 30, 50):
            due = rows[f"n_due_at_{threshold}"].to_numpy(dtype=int)
            not_due = rows[f"n_not_due_at_{threshold}"].to_numpy(dtype=int)
            if len(np.unique(due)) != 1 or len(np.unique(not_due)) != 1:
                raise RuntimeError(
                    "Maintenance strata disagree across methods for "
                    f"{dataset} at threshold {threshold}"
                )
            lines.append(
                f"{dataset} & {threshold} & {int(due[0])} & "
                f"{int(not_due[0])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    lines.extend(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Conditional delayed- and premature-action rates for all "
            "18 locked final methods at the protocol-specified threshold "
            "grid. D and P denote delayed and premature action, respectively. "
            "The thresholds are sensitivity-analysis proxies, not calibrated "
            "maintenance limits. Rates describe the finite benchmark and are "
            "not population-level risk estimates.}",
            "\\label{sup:tab:maintenance-thresholds}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{@{}llrrrrrrr@{}}",
            "\\toprule",
            "Subset & Selection & $c$ & D@20 & P@20 & D@30 & P@30 & "
            "D@50 & P@50 \\\\",
            "\\midrule",
        ]
    )
    for dataset in DATASETS:
        for method in final_methods_for_dataset(dataset):
            row = index.loc[(dataset, method)]
            lines.append(
                f"{dataset} & {LABELS[method]} & {int(row['train_cap'])} & "
                f"{f3(row['delayed_action_rate_at_20'])} & "
                f"{f3(row['premature_action_rate_at_20'])} & "
                f"{f3(row['delayed_action_rate_at_30'])} & "
                f"{f3(row['premature_action_rate_at_30'])} & "
                f"{f3(row['delayed_action_rate_at_50'])} & "
                f"{f3(row['premature_action_rate_at_50'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_saturation_table(finals):
    index = finals.set_index(["dataset", "method"])
    lines = [
        "",
        "\\section{Locked test-set saturation diagnostics}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Test-set target saturation and exposure for all locked "
        "cap-grid and TPE comparisons. The saturated fraction $q_{\\rm test}(c)$ "
        "counts standard-test engines with original RUL above $c$; "
        "$E_{r,\\rm test}(c)$ and $E_{\\rm test}(c)$ are the reference- and "
        "uncapped-scale exposure bounds in cycles. The final two columns give "
        "each squared bound as a fraction of its aligned observed "
        "candidate-ceiling MSE gap.}",
        "\\label{sup:tab:test-saturation}",
        "\\scriptsize",
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Subset & Selection & $c$ & $q_{\\rm test}$ (\\%) & "
        "$E_{r,\\rm test}$ & $E_{\\rm test}$ & "
        "$E_r^2/\\Delta\\mathrm{MSE}_r$ & "
        "$E^2/\\Delta\\mathrm{MSE}_{\\rm orig}$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for method in final_methods_for_dataset(dataset):
            row = index.loc[(dataset, method)]
            lines.append(
                f"{dataset} & {LABELS[method]} & {int(row['train_cap'])} & "
                f"{100.0 * float(row['saturation_fraction']):.1f} & "
                f"{f2(row['reference_exposure_bound'])} & "
                f"{f2(row['exposure_bound'])} & "
                f"{f3_optional(row['reference_bound_fraction_of_hidden_gap'])} & "
                f"{f3_optional(row['bound_fraction_of_hidden_gap'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def f1(value):
    return f"{float(value):.1f}"


def four_subset_phrase(count):
    if count == 0:
        return "none of the four subsets"
    if count == 4:
        return "all four subsets"
    return f"{count} of four subsets"


def result_text_fragments(decisions, tpe, pairs, finals, decomposition):
    decision_index = decisions.set_index(["dataset", "objective"])
    self_caps = [
        int(decision_index.loc[(dataset, "self_scaled"), "selected_cap"])
        for dataset in DATASETS
    ]
    fixed_caps = [
        int(decision_index.loc[(dataset, "fixed_reference"), "selected_cap"])
        for dataset in DATASETS
    ]
    different_caps = sum(
        left != right for left, right in zip(self_caps, fixed_caps, strict=True)
    )
    self_lowest_count = sum(cap == 90 for cap in self_caps)
    grid_pairs = (
        pairs[pairs["comparison"] == "fixed_vs_self_cap_grid"]
        .set_index("dataset")
        .loc[DATASETS]
    )
    grid_deltas = grid_pairs[
        "paired_engine_rmse_difference"
    ].to_numpy(dtype=float)
    lower_count = int(np.sum(grid_deltas < -1e-12))
    higher_count = int(np.sum(grid_deltas > 1e-12))
    tied_count = len(DATASETS) - lower_count - higher_count
    fixed125_pairs = (
        pairs[pairs["comparison"] == "fixed_cap_grid_vs_fixed125"]
        .set_index("dataset")
        .loc[DATASETS]
    )
    fixed125_deltas = fixed125_pairs[
        "paired_engine_rmse_difference"
    ].to_numpy(dtype=float)
    fixed125_lower_count = int(np.sum(fixed125_deltas < -1e-12))
    fixed125_higher_count = int(np.sum(fixed125_deltas > 1e-12))
    fixed125_tied_count = (
        len(DATASETS) - fixed125_lower_count - fixed125_higher_count
    )
    final_index = finals.set_index(["dataset", "method"])
    delayed_deltas = np.asarray(
        [
            float(
                final_index.loc[
                    (dataset, "cap_grid_fixed_reference"),
                    "delayed_action_rate_at_30",
                ]
            )
            - float(
                final_index.loc[
                    (dataset, "cap_grid_self_scaled"),
                    "delayed_action_rate_at_30",
                ]
            )
            for dataset in DATASETS
        ]
    )
    premature_deltas = np.asarray(
        [
            float(
                final_index.loc[
                    (dataset, "cap_grid_fixed_reference"),
                    "premature_action_rate_at_30",
                ]
            )
            - float(
                final_index.loc[
                    (dataset, "cap_grid_self_scaled"),
                    "premature_action_rate_at_30",
                ]
            )
            for dataset in DATASETS
        ]
    )
    delayed_lower_count = int(np.sum(delayed_deltas < -1e-12))
    delayed_higher_count = int(np.sum(delayed_deltas > 1e-12))
    delayed_tied_count = (
        len(DATASETS) - delayed_lower_count - delayed_higher_count
    )
    pool_index = {
        (
            row["dataset"],
            row["checkpoint_objective"],
            row["target_scale"],
            row["output_support"],
        ): row["selected_cap"]
        for row in checkpoint_pool_selection_rows(decomposition)
    }
    target_scale_changes = sum(
        pool_index[(dataset, objective, "candidate", support)]
        != pool_index[(dataset, objective, "reference", support)]
        for dataset in DATASETS
        for objective in ("self_scaled", "fixed_reference")
        for support in ("candidate", "reference")
    )
    output_support_changes = sum(
        pool_index[(dataset, objective, target, "candidate")]
        != pool_index[(dataset, objective, target, "reference")]
        for dataset in DATASETS
        for objective in ("self_scaled", "fixed_reference")
        for target in ("candidate", "reference")
    )

    abstract = (
        "% Generated by generate_manuscript_assets.py; do not edit.\n"
        "Self-scaled chose the lowest cap on "
        f"{four_subset_phrase(self_lowest_count)}; fixed-reference caps: "
        + "/".join(str(cap) for cap in fixed_caps)
        + ". Their reference-scale RMSE difference ranged from "
        f"{f2(np.min(grid_deltas))} to {f2(np.max(grid_deltas))}. "
        "Fixed-reference-minus-self RMSE differences were "
        f"negative/positive/numerically zero on "
        f"{lower_count}/{higher_count}/{tied_count} "
        "subsets; against fixed-$125$, the counts were "
        f"{fixed125_lower_count}/{fixed125_higher_count}/"
        f"{fixed125_tied_count}. Its 30-cycle delayed-action-rate differences "
        f"were negative/positive/numerically zero on {delayed_lower_count}/"
        f"{delayed_higher_count}/{delayed_tied_count} subsets. In a post-hoc "
        "archived-checkpoint-pool reranking, target-scale changes altered the "
        f"selected cap in {target_scale_changes}/16 paired contrasts, whereas "
        "output-support changes alone did so in "
        f"{output_support_changes}/16.\n"
    )
    cap_rows = ", ".join(
        f"{dataset}: {self_cap}/{fixed_cap}"
        for dataset, self_cap, fixed_cap in zip(
            DATASETS, self_caps, fixed_caps, strict=True
        )
    )
    delta_rows = "; ".join(
        f"{dataset} {f2(row['paired_engine_rmse_difference'])} "
        f"[{f2(row['paired_engine_bootstrap_95ci_low'])}, "
        f"{f2(row['paired_engine_bootstrap_95ci_high'])}]"
        for dataset, row in grid_pairs.iterrows()
    )
    fixed125_delta_rows = "; ".join(
        f"{dataset} {f2(row['paired_engine_rmse_difference'])} "
        f"[{f2(row['paired_engine_bootstrap_95ci_low'])}, "
        f"{f2(row['paired_engine_bootstrap_95ci_high'])}]"
        for dataset, row in fixed125_pairs.iterrows()
    )
    tpe_index = tpe.set_index(["dataset", "objective"])
    tpe_caps = ", ".join(
        f"{dataset}: "
        f"{int(tpe_index.loc[(dataset, 'self_scaled'), 'best_train_cap'])}/"
        f"{int(tpe_index.loc[(dataset, 'fixed_reference'), 'best_train_cap'])}/"
        f"{int(tpe_index.loc[(dataset, 'fixed_cap_125'), 'best_train_cap'])}"
        for dataset in ("FD001", "FD002")
    )
    tpe_pairs = (
        pairs[pairs["comparison"] == "fixed_vs_self_tpe"]
        .set_index("dataset")
        .loc[["FD001", "FD002"]]
    )
    tpe_delta_rows = ", ".join(
        f"{dataset} {f2(row['paired_engine_rmse_difference'])}"
        for dataset, row in tpe_pairs.iterrows()
    )
    delayed_delta_rows = ", ".join(
        f"{dataset} {f3(value)}"
        for dataset, value in zip(DATASETS, delayed_deltas, strict=True)
    )
    premature_delta_rows = ", ".join(
        f"{dataset} {f3(value)}"
        for dataset, value in zip(DATASETS, premature_deltas, strict=True)
    )
    narrative = "\n".join(
        [
            "% Generated by generate_manuscript_assets.py; do not edit.",
            "\\paragraph{Cap-grid selection on a common reporting scale} "
            "The self-scaled/fixed-reference "
            "cap-grid selections were "
            + cap_rows
            + f", differing on {different_caps} of four subsets. On the "
            "common target and reporting clip, fixed-reference minus "
            "self-scaled median-ensemble RMSE differences (95\\% paired "
            "engine-bootstrap intervals) were "
            + delta_rows
            + f". The differences were negative on {lower_count}, "
            f"positive on {higher_count}, and numerically zero on "
            f"{tied_count} subsets "
            "at a numerical tolerance of $10^{-12}$.",
            "\\paragraph{Comparison with the fixed-cap benchmark control} "
            "Against the practical fixed-$c=125$ control, fixed-reference "
            "minus control median-ensemble RMSE differences (95\\% paired "
            "engine-bootstrap intervals) were "
            + fixed125_delta_rows
            + f". The differences were negative on "
            f"{fixed125_lower_count}, positive on "
            f"{fixed125_higher_count}, and numerically zero on "
            f"{fixed125_tied_count} subsets. This control is the "
            "default benchmark practice when cap selection is not itself "
            "the research question.",
            "\\paragraph{Checkpoint-pool mechanism diagnostic} In a post-hoc "
            "reranking of the five caps within each archived checkpoint pool, "
            "changing the target scale altered the selected cap in "
            f"{target_scale_changes}/16 paired contrasts, whereas changing "
            "only the output support altered it in "
            f"{output_support_changes}/16. This diagnostic localizes the "
            "observed cap-ranking change within the archived candidates; it "
            "does not reconstruct unarchived epoch histories or replace an "
            "independently executed target-only selection arm.",
            "\\paragraph{Maintenance-proxy sensitivity} At the "
            "protocol-specified 30-cycle maintenance-proxy threshold, "
            "fixed-reference minus self-scaled delayed-action rate "
            "differences were "
            + delayed_delta_rows
            + "; premature-action rate differences were "
            + premature_delta_rows
            + f". Delayed-action-rate differences were negative on "
            f"{delayed_lower_count}, positive on {delayed_higher_count}, and "
            f"numerically zero on {delayed_tied_count} subsets. These "
            "conditional rates "
            "are sensitivity-analysis proxies, not calibrated maintenance "
            "costs.",
            "\\paragraph{Equal-trial TPE check and frozen-winner sensitivity} "
            "The matched 12-trial TPE self/fixed-reference/fixed-cap-125 "
            "winners used caps "
            + tpe_caps
            + ". The main table reports both the candidate's own validation "
            "RMSE and its fixed-reference validation RMSE, so any apparent "
            "gain caused by a changed yardstick remains visible. Their "
            "fixed-reference minus self-scaled common-target "
            "ensemble RMSE differences were "
            + tpe_delta_rows
            + ". These plan-bound values are descriptive because of the "
            "disclosed prior C-MAPSS test contact; the complete seed "
            "dispersion, bootstrap summaries, and frozen-winner split/seed "
            "sensitivity checks appear "
            "in the tables and Supplementary Material.",
            "",
        ]
    )
    cover = (
        f"In the locked rerun, the cap-grid objectives selected different "
        f"caps on {different_caps} of four C-MAPSS subsets. On the common "
        f"reporting target and reporting clip, fixed-reference selection "
        f"had negative/positive/numerically zero RMSE differences on "
        f"{lower_count}/{higher_count}/{tied_count} subsets; relative to the "
        f"fixed-cap-125 control, the corresponding counts were "
        f"{fixed125_lower_count}/{fixed125_higher_count}/"
        f"{fixed125_tied_count}. The post-hoc archived-checkpoint-pool "
        f"diagnostic changed cap in {target_scale_changes}/16 target-scale "
        f"contrasts and {output_support_changes}/16 output-support-only "
        f"contrasts. At the predeclared 30-cycle proxy "
        f"threshold, its delayed-action-rate differences were "
        f"negative/positive/numerically zero on "
        f"{delayed_lower_count}/{delayed_higher_count}/"
        f"{delayed_tied_count} subsets. These are descriptive plan-bound "
        f"comparisons; the "
        f"manuscript reports their paired engine-bootstrap intervals and "
        f"the matched 12-trial TPE mechanism check."
    )
    highlights = [
        f"\\item The objectives selected different caps on "
        f"{different_caps}/4 C-MAPSS subsets.",
        f"\\item Fixed-reference minus self RMSE was negative/positive/zero on "
        f"{lower_count}/{higher_count}/{tied_count} subsets.",
    ]
    graphical = (
        "% Generated by generate_manuscript_assets.py; do not edit.\n"
        f"\\def\\GraphicalResultLineOne{{Different caps on "
        f"{different_caps}/4 subsets}}\n"
        f"\\def\\GraphicalResultLineTwo{{RMSE $\\Delta$: "
        f"{lower_count}/{higher_count}/{tied_count} "
        "neg./pos./zero}}\n"
    )
    return abstract, narrative, cover, highlights, graphical


def update_cover_result_block(path: Path, paragraph: str) -> None:
    start = "<!-- GENERATED_RESULTS_START -->"
    end = "<!-- GENERATED_RESULTS_END -->"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL
    )
    if len(pattern.findall(text)) != 1:
        raise RuntimeError("Cover-letter generated-results block is invalid")
    replacement = f"{start}\n\n{paragraph}\n\n{end}"
    atomic_text(path, pattern.sub(replacement, text))


def update_highlight_block(path: Path, items: list[str]) -> None:
    start = "% GENERATED_HIGHLIGHTS_START"
    end = "% GENERATED_HIGHLIGHTS_END"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL
    )
    if len(pattern.findall(text)) != 1:
        raise RuntimeError("Highlights generated-results block is invalid")
    if len(items) != 2 or any(not item.startswith("\\item ") for item in items):
        raise RuntimeError("Exactly two generated highlight items are required")
    replacement = f"{start}\n" + "\n".join(items) + f"\n{end}"
    atomic_text(path, pattern.sub(lambda _: replacement, text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2\summaries"),
    )
    parser.add_argument(
        "--manuscript_dir",
        type=Path,
        default=Path(r"D:\EAAI\Fixed_Reference_RUL_Model_Selection\manuscript"),
    )
    parser.add_argument(
        "--assets_only",
        action="store_true",
        help=(
            "Generate standalone figures and LaTeX result fragments without "
            "requiring cover-letter or Highlights templates."
        ),
    )
    args = parser.parse_args()
    summary = args.summary_dir.resolve()
    manuscript = args.manuscript_dir.resolve()
    figures = manuscript / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    status = validate_summary_provenance(summary)
    validation = load_required(summary / "validation_long.csv")
    cap_summary = load_required(summary / "cap_summary.csv")
    aggregation = load_required(
        summary / "aggregation_sensitivity.csv"
    )
    stability = load_required(summary / "stability_long.csv")
    decisions = load_required(summary / "locked_decisions.csv")
    tpe = load_required(summary / "tpe_locks.csv")
    tpe_robustness = load_required(
        summary / "tpe_robustness.csv"
    )
    decomposition = load_required(
        summary / "target_support_decomposition.csv"
    )
    finals = load_required(summary / "final_test.csv")
    pairs = load_required(summary / "paired_tests.csv")
    require_unique(
        cap_summary, ["dataset", "objective", "cap"], 40, "cap summary"
    )
    require_unique(
        aggregation,
        ["dataset", "objective", "aggregation_scheme"],
        32,
        "aggregation sensitivity",
    )
    require_unique(
        stability,
        ["dataset", "objective", "cap", "model_seed"],
        80,
        "seed stability",
    )
    require_unique(
        decisions, ["dataset", "objective"], 12, "locked decisions"
    )
    require_unique(tpe, ["dataset", "objective"], 6, "TPE locks")
    require_unique(
        tpe_robustness,
        ["dataset", "objective"],
        6,
        "TPE robustness",
    )
    require_unique(
        decomposition,
        [
            "dataset",
            "checkpoint_objective",
            "cap",
            "target_scale",
            "output_support",
        ],
        160,
        "target/output-ceiling decomposition",
    )
    require_unique(finals, ["dataset", "method"], 18, "final tests")
    require_unique(pairs, ["dataset", "comparison"], 12, "paired tests")
    require_exact_key_set(
        cap_summary,
        ["dataset", "objective", "cap"],
        {
            (dataset, objective, cap)
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
            for cap in (90, 100, 110, 115, 125)
        },
        "Cap summary",
    )
    require_exact_key_set(
        aggregation,
        ["dataset", "objective", "aggregation_scheme"],
        {
            (dataset, objective, scheme)
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
            for scheme in (
                "split_equal",
                "run_equal",
                "primary_three_seed_mean",
                "seed42_three_split_mean",
            )
        },
        "Aggregation sensitivity",
    )
    require_exact_key_set(
        stability,
        ["dataset", "objective", "cap", "model_seed"],
        {
            (dataset, objective, cap, seed)
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
            for cap in (90, 100, 110, 115, 125)
            for seed in (43, 44)
        },
        "Seed stability",
    )
    require_exact_key_set(
        decisions,
        ["dataset", "objective"],
        {
            (dataset, objective)
            for dataset in DATASETS
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            )
        },
        "Locked decisions",
    )
    require_exact_key_set(
        tpe,
        ["dataset", "objective"],
        {
            (dataset, objective)
            for dataset in ("FD001", "FD002")
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            )
        },
        "TPE locks",
    )
    require_exact_key_set(
        tpe_robustness,
        ["dataset", "objective"],
        {
            (dataset, objective)
            for dataset in ("FD001", "FD002")
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            )
        },
        "TPE robustness",
    )
    require_exact_key_set(
        decomposition,
        [
            "dataset",
            "checkpoint_objective",
            "cap",
            "target_scale",
            "output_support",
        ],
        {
            (dataset, objective, cap, target, support)
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
            for cap in (90, 100, 110, 115, 125)
            for target in ("candidate", "reference")
            for support in ("candidate", "reference")
        },
        "Target/output-ceiling decomposition",
    )
    expected_final_keys = {
        *((dataset, method) for dataset in DATASETS for method in METHODS),
        *((dataset, f"tpe_{objective}")
          for dataset in ("FD001", "FD002")
          for objective in (
              "self_scaled",
              "fixed_reference",
              "fixed_cap_125",
          )),
    }
    if set(
        zip(finals["dataset"], finals["method"], strict=True)
    ) != expected_final_keys:
        raise RuntimeError("Final-test method matrix is invalid")
    require_exact_key_set(
        pairs,
        ["dataset", "comparison"],
        {
            *((dataset, "fixed_vs_self_cap_grid") for dataset in DATASETS),
            *((dataset, "fixed_cap_grid_vs_fixed125") for dataset in DATASETS),
            *((dataset, "fixed_vs_self_tpe")
              for dataset in ("FD001", "FD002")),
            *((dataset, "fixed_reference_tpe_vs_fixed_cap_125_tpe")
              for dataset in ("FD001", "FD002")),
        },
        "Paired comparisons",
    )
    validate_reportable_metrics(finals, pairs, validation)
    validation_figure(
        cap_summary, figures / "fig_confirmatory_validation.pdf"
    )
    final_rmse_figure(
        finals, figures / "fig_confirmatory_test_rmse.pdf"
    )
    provenance = (
        "% Summary campaign: "
        f"{status['campaign_id']}; final summary SHA-256: "
        f"{status['final_test_summary_sha256']}\n"
    )
    atomic_text(
        manuscript / "generated_confirmatory_tables.tex",
        provenance + latex_tables(decisions, tpe, finals, pairs),
    )
    atomic_text(
        manuscript / "generated_supplementary_tables.tex",
        provenance
        + supplementary_pair_table(pairs)
        + supplementary_individual_table(finals)
        + supplementary_tpe_robustness_table(tpe_robustness)
        + supplementary_aggregation_sensitivity_table(aggregation)
        + supplementary_validation_two_by_two_table(decomposition)
        + supplementary_checkpoint_pool_selection_table(decomposition)
        + supplementary_validation_saturation_table(validation)
        + supplementary_two_by_two_contrast_table(finals)
        + supplementary_maintenance_threshold_table(finals)
        + supplementary_saturation_table(finals),
    )
    (
        abstract_result,
        results_narrative,
        cover_result,
        generated_highlights,
        graphical_result,
    ) = result_text_fragments(
        decisions, tpe, pairs, finals, decomposition
    )
    atomic_text(
        manuscript / "generated_abstract_result.tex", abstract_result
    )
    atomic_text(
        manuscript / "generated_results_narrative.tex", results_narrative
    )
    atomic_text(
        manuscript / "generated_graphical_result.tex", graphical_result
    )
    if not args.assets_only:
        update_cover_result_block(
            manuscript / "cover_letter.md", cover_result
        )
        update_highlight_block(
            manuscript / "highlights.tex", generated_highlights
        )
    print("Generated locked manuscript assets.")


if __name__ == "__main__":
    main()
