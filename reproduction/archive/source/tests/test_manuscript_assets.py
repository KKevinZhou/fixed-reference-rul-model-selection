import hashlib
import json
from pathlib import Path

import pandas as pd

from generate_manuscript_assets import (
    DATASETS,
    checkpoint_pool_selection_rows,
    latex_tables,
    result_text_fragments,
    supplementary_individual_table,
    supplementary_aggregation_sensitivity_table,
    supplementary_checkpoint_pool_selection_table,
    supplementary_maintenance_threshold_table,
    supplementary_pair_table,
    supplementary_saturation_table,
    supplementary_tpe_robustness_table,
    supplementary_two_by_two_contrast_table,
    supplementary_validation_saturation_table,
    supplementary_validation_two_by_two_table,
    two_by_two_mse_contrasts,
    update_cover_result_block,
    update_highlight_block,
    validate_summary_provenance,
)


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER_SHA256 = hashlib.sha256(
    (ARTIFACT_ROOT / "summarize_confirmatory.py").read_bytes()
).hexdigest()
ASSET_GENERATOR_SHA256 = hashlib.sha256(
    (ARTIFACT_ROOT / "generate_manuscript_assets.py").read_bytes()
).hexdigest()


def test_generated_tables_cover_predeclared_contrasts():
    decisions = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "objective": objective,
                "selected_cap": 125 if objective == "fixed_reference" else 90,
                "selection_value": 10.0,
            }
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
        ]
    )
    finals = []
    methods = (
        "cap_grid_self_scaled",
        "cap_grid_fixed_reference",
        "fixed_cap_125",
    )
    for dataset in DATASETS:
        for method in methods:
            finals.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "train_cap": 125,
                    "own_rmse_ensemble": 9.0,
                    "candidate_target_common_support_rmse_ensemble": 9.5,
                    "candidate_support_reference_rmse_ensemble": 10.0,
                    "reference_rmse_ensemble": 10.0,
                    "nasa_score_reference_ensemble": 100.0,
                    "reference_rmse_mean": 10.0,
                    "reference_rmse_std": 0.5,
                    "reference_rmse_bootstrap_low": 8.0,
                    "reference_rmse_bootstrap_high": 12.0,
                    "mean_error_ensemble": 0.1,
                    "late_prediction_fraction": 0.5,
                    "n_due_at_20": 20,
                    "n_not_due_at_20": 80,
                    "delayed_action_rate_at_20": 0.3,
                    "premature_action_rate_at_20": 0.05,
                    "n_due_at_30": 30,
                    "n_not_due_at_30": 70,
                    "delayed_action_rate_at_30": 0.2,
                    "premature_action_rate_at_30": 0.1,
                    "n_due_at_50": 50,
                    "n_not_due_at_50": 50,
                    "delayed_action_rate_at_50": 0.1,
                    "premature_action_rate_at_50": 0.2,
                    "saturation_fraction": 0.2,
                    "exposure_bound": 15.0,
                    "bound_fraction_of_hidden_gap": 0.8,
                    "reference_exposure_bound": 12.0,
                    "reference_bound_fraction_of_hidden_gap": 0.7,
                }
            )
    tpe = []
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            tpe.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "best_train_cap": 125,
                    "best_lr": 5e-4,
                    "best_batch": 128,
                    "best_huber_delta": 1.0,
                    "best_value": 10.0,
                    "best_own_rmse": 9.0,
                    "best_reference_rmse": 10.0,
                }
            )
            finals.append(
                {
                    "dataset": dataset,
                    "method": f"tpe_{objective}",
                    "train_cap": 125,
                    "own_rmse_ensemble": 9.0,
                    "candidate_target_common_support_rmse_ensemble": 9.5,
                    "candidate_support_reference_rmse_ensemble": 10.0,
                    "reference_rmse_ensemble": 10.0,
                    "nasa_score_reference_ensemble": 100.0,
                    "reference_rmse_mean": 10.0,
                    "reference_rmse_std": 0.5,
                    "reference_rmse_bootstrap_low": 8.0,
                    "reference_rmse_bootstrap_high": 12.0,
                    "mean_error_ensemble": 0.1,
                    "late_prediction_fraction": 0.5,
                    "n_due_at_20": 20,
                    "n_not_due_at_20": 80,
                    "delayed_action_rate_at_20": 0.3,
                    "premature_action_rate_at_20": 0.05,
                    "n_due_at_30": 30,
                    "n_not_due_at_30": 70,
                    "delayed_action_rate_at_30": 0.2,
                    "premature_action_rate_at_30": 0.1,
                    "n_due_at_50": 50,
                    "n_not_due_at_50": 50,
                    "delayed_action_rate_at_50": 0.1,
                    "premature_action_rate_at_50": 0.2,
                    "saturation_fraction": 0.2,
                    "exposure_bound": 15.0,
                    "bound_fraction_of_hidden_gap": 0.8,
                    "reference_exposure_bound": 12.0,
                    "reference_bound_fraction_of_hidden_gap": 0.7,
                }
            )
    pairs = []
    for dataset in DATASETS:
        labels = [
            "fixed_vs_self_cap_grid",
            "fixed_cap_grid_vs_fixed125",
        ]
        if dataset in {"FD001", "FD002"}:
            labels.extend(
                [
                    "fixed_vs_self_tpe",
                    "fixed_reference_tpe_vs_fixed_cap_125_tpe",
                ]
            )
        for label in labels:
            pairs.append(
                {
                    "dataset": dataset,
                    "comparison": label,
                    "mean_difference": -1.0,
                    "paired_engine_rmse_difference": -1.0,
                    "paired_engine_bootstrap_95ci_low": -2.0,
                    "paired_engine_bootstrap_95ci_high": 0.0,
                        "bootstrap_fraction_negative": 0.9,
                        "paired_difference_values": json.dumps(
                            [-2.0, -1.5, -1.0, -0.5, 0.0]
                        ),
                        "wilcoxon_holm_primary_pvalue": (
                        0.25
                        if label == "fixed_vs_self_cap_grid"
                        else None
                    ),
                    "wilcoxon_holm_secondary_pvalue": (
                        None
                        if label == "fixed_vs_self_cap_grid"
                        else 0.5
                    ),
                    "wilcoxon_holm_all12_pvalue": 0.75,
                }
            )
    pair_frame = pd.DataFrame(pairs)
    main = latex_tables(
        decisions,
        pd.DataFrame(tpe),
        pd.DataFrame(finals),
        pair_frame,
    )
    supplement = supplementary_pair_table(pair_frame)
    individual = supplementary_individual_table(pd.DataFrame(finals))
    contrasts = supplementary_two_by_two_contrast_table(
        pd.DataFrame(finals)
    )
    maintenance = supplementary_maintenance_threshold_table(
        pd.DataFrame(finals)
    )
    diagnostics = supplementary_saturation_table(pd.DataFrame(finals))
    assert "\\label{tab:primary-paired}" in main
    assert "$T_c/S_r$" in main
    assert "$\\Delta_{\\rm int}$" in main
    assert "-9.25" in main
    assert "Ref.-scale NASA" in main
    assert "Own val. & Ref. val. & Test RMSE" in main
    assert "\\label{sup:tab:all-paired}" in supplement
    assert supplement.count("TPE: ref.--self") == 4
    assert supplement.count("TPE: ref.--fixed-125") == 4
    assert "Holm$_{\\rm fam}$" in supplement
    assert "Bootstrap fraction $\\Delta<0$" in supplement
    assert "\\label{sup:tab:individual-test}" in individual
    assert "[8.00, 12.00]" in individual
    assert "\\label{sup:tab:two-by-two-contrasts}" in contrasts
    assert contrasts.count("TPE self-scaled") == 2
    assert "-9.25" in contrasts
    assert "\\label{sup:tab:maintenance-thresholds}" in maintenance
    assert "\\label{sup:tab:maintenance-denominators}" in maintenance
    assert "D@20" in maintenance
    assert maintenance.count("TPE self-scaled") == 2
    assert "\\label{sup:tab:test-saturation}" in diagnostics
    assert diagnostics.count("Self-scaled") == 4


def test_validation_saturation_table_collapses_objective_duplicates():
    rows = []
    for dataset in DATASETS:
        for cap in (90, 100, 110, 115, 125):
            for split_index, split in enumerate(
                (20260724, 20260725, 20260726)
            ):
                for objective in ("self_scaled", "fixed_reference"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_seed": split,
                            "cap": cap,
                            "objective": objective,
                            "saturation_fraction": 0.1 + 0.01 * split_index,
                            "reference_exposure_bound": 2.0 + split_index,
                            "exposure_bound": 3.0 + split_index,
                        }
                    )
    table = supplementary_validation_saturation_table(pd.DataFrame(rows))
    assert "\\label{sup:tab:validation-saturation}" in table
    assert table.count("FD001 &") == 5
    assert "$q_{\\rm val}$" in table


def test_validation_two_by_two_table_consumes_all_160_cells():
    rows = []
    cell_mse = {
        ("candidate", "candidate"): 9.0,
        ("candidate", "reference"): 9.5,
        ("reference", "candidate"): 10.0,
        ("reference", "reference"): 10.0,
    }
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            for cap in (90, 100, 110, 115, 125):
                for (target, support), value in cell_mse.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "checkpoint_objective": objective,
                            "cap": cap,
                            "target_scale": target,
                            "output_support": support,
                            "split_values": json.dumps(
                                [value, value, value]
                            ),
                        }
                    )
    assert len(rows) == 160
    table = supplementary_validation_two_by_two_table(
        pd.DataFrame(rows)
    )
    assert "\\label{sup:tab:validation-two-by-two}" in table
    assert table.count("FD001 &") == 10
    assert table.count("-0.50") == 40


def test_checkpoint_pool_selection_table_reranks_caps_with_stable_ties():
    rows = []
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            for cap in (90, 100, 110, 115, 125):
                for target in ("candidate", "reference"):
                    for support in ("candidate", "reference"):
                        preferred = 90 if target == "candidate" else 115
                        rows.append(
                            {
                                "dataset": dataset,
                                "checkpoint_objective": objective,
                                "cap": cap,
                                "target_scale": target,
                                "output_support": support,
                                "mean_across_splits": (
                                    100.0 + abs(cap - preferred)
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    selections = checkpoint_pool_selection_rows(frame)
    table = supplementary_checkpoint_pool_selection_table(frame)
    assert len(selections) == 32
    assert "\\label{sup:tab:checkpoint-pool-selection}" in table
    assert table.count("FD001 &") == 2
    assert table.count("90 (10.00)") == 16
    assert table.count("115 (10.00)") == 16
    assert "does not reconstruct unarchived epoch histories" in table


def test_aggregation_sensitivity_table_consumes_all_32_decisions():
    rows = [
        {
            "dataset": dataset,
            "objective": objective,
            "aggregation_scheme": scheme,
            "selected_cap": 125 if "split" in scheme else 115,
        }
        for dataset in DATASETS
        for objective in ("self_scaled", "fixed_reference")
        for scheme in (
            "split_equal",
            "run_equal",
            "primary_three_seed_mean",
            "seed42_three_split_mean",
        )
    ]
    table = supplementary_aggregation_sensitivity_table(
        pd.DataFrame(rows)
    )
    assert "\\label{sup:tab:aggregation-sensitivity}" in table
    assert table.count("FD001 &") == 2
    assert "Primary 3-seed" in table


def test_tpe_robustness_table_consumes_all_six_frozen_winners():
    rows = [
        {
            "dataset": dataset,
            "objective": objective,
            "selection_value": 10.0,
            "primary_seed43_value": 10.1,
            "primary_seed44_value": 10.2,
            "secondary_split20260725_seed42": 10.3,
            "secondary_split20260726_seed42": 10.4,
        }
        for dataset in ("FD001", "FD002")
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        )
    ]
    table = supplementary_tpe_robustness_table(pd.DataFrame(rows))
    assert "\\label{sup:tab:tpe-robustness}" in table
    assert table.count("FD001 &") == 3
    assert table.count("10.40") == 6


def test_two_by_two_mse_contrasts_are_algebraically_consistent():
    values = two_by_two_mse_contrasts(
        {
            "own_rmse_ensemble": 9.0,
            "candidate_target_common_support_rmse_ensemble": 9.5,
            "candidate_support_reference_rmse_ensemble": 10.0,
            "reference_rmse_ensemble": 10.0,
        }
    )
    assert values["Lcc"] == 81.0
    assert values["Lcr"] == 90.25
    assert values["Lrc"] == 100.0
    assert values["Lrr"] == 100.0
    assert values["delta_t_r"] - values["delta_t_c"] == -9.25
    assert values["delta_s_r"] - values["delta_s_c"] == -9.25
    assert values["delta_int"] == -9.25


def test_result_fragments_and_cover_block_are_generated_idempotently(tmp_path):
    decisions = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "objective": objective,
                "selected_cap": (
                    90
                    if objective == "self_scaled"
                    else 125
                ),
            }
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
        ]
    )
    tpe = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "objective": objective,
                "best_train_cap": (
                    125 if objective == "fixed_cap_125" else 100
                ),
            }
            for dataset in ("FD001", "FD002")
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            )
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "comparison": comparison,
                "paired_engine_rmse_difference": (
                    -1.0 if dataset != "FD004" else 0.5
                ),
                "paired_engine_bootstrap_95ci_low": -2.0,
                "paired_engine_bootstrap_95ci_high": 1.0,
            }
            for comparison, datasets in (
                ("fixed_vs_self_cap_grid", DATASETS),
                ("fixed_cap_grid_vs_fixed125", DATASETS),
                ("fixed_vs_self_tpe", ("FD001", "FD002")),
            )
            for dataset in datasets
        ]
    )
    finals = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "method": method,
                "delayed_action_rate_at_30": (
                    0.1
                    if method == "cap_grid_fixed_reference"
                    and dataset != "FD004"
                    else 0.2
                ),
                "premature_action_rate_at_30": (
                    0.2
                    if method == "cap_grid_fixed_reference"
                    else 0.1
                ),
            }
            for dataset in DATASETS
            for method in (
                "cap_grid_self_scaled",
                "cap_grid_fixed_reference",
            )
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "checkpoint_objective": objective,
                "cap": cap,
                "target_scale": target,
                "output_support": support,
                "mean_across_splits": (
                    100.0
                    + abs(
                        cap
                        - (90 if target == "candidate" else 115)
                    )
                ),
            }
            for dataset in DATASETS
            for objective in ("self_scaled", "fixed_reference")
            for cap in (90, 100, 110, 115, 125)
            for target in ("candidate", "reference")
            for support in ("candidate", "reference")
        ]
    )
    abstract, narrative, cover, highlights, graphical = result_text_fragments(
        decisions, tpe, pairs, finals, decomposition
    )
    assert "lowest cap on all four subsets" in abstract
    assert "fixed-reference caps: 125/125/125/125" in abstract
    assert "RMSE difference ranged from -1.00 to 0.50" in abstract
    assert "negative/positive/numerically zero on 3/1/0 subsets" in abstract
    assert "Cap-grid selection on a common reporting scale" in narrative
    assert "Comparison with the fixed-cap benchmark control" in narrative
    assert "Equal-trial TPE check and frozen-winner sensitivity" in narrative
    assert "FD001: 90/125" in narrative
    assert (
        "30-cycle delayed-action-rate differences "
        "were negative/positive/numerically zero on 3/0/1"
        in abstract
    )
    assert "Delayed-action-rate differences were negative on 3" in narrative
    assert "practical fixed-$c=125$ control" in narrative
    assert "target scale altered the selected cap in 16/16" in narrative
    assert "output support altered it in 0/16" in narrative
    assert "both the candidate's own validation RMSE" in narrative
    assert "against fixed-$125$, the counts were 3/1/0" in abstract
    assert "negative/positive/numerically zero RMSE differences on 3/1/0" in cover
    assert "fixed-cap-125 control" in cover
    assert "16/16 target-scale contrasts" in cover
    assert "0/16 output-support-only contrasts" in cover
    assert "$" not in cover
    assert len(highlights) == 2
    assert "negative/positive/zero on 3/1/0 subsets" in highlights[1]
    assert "Different caps on 4/4 subsets" in graphical
    assert "3/1/0 neg./pos./zero" in graphical

    cover_path = tmp_path / "cover_letter.md"
    cover_path.write_text(
        "Before\n<!-- GENERATED_RESULTS_START -->\nold\n"
        "<!-- GENERATED_RESULTS_END -->\nAfter\n",
        encoding="utf-8",
    )
    update_cover_result_block(cover_path, cover)
    first = cover_path.read_text(encoding="utf-8")
    update_cover_result_block(cover_path, cover)
    assert cover_path.read_text(encoding="utf-8") == first
    assert cover in first

    highlight_path = tmp_path / "highlights.tex"
    highlight_path.write_text(
        "Before\n% GENERATED_HIGHLIGHTS_START\nold\n"
        "% GENERATED_HIGHLIGHTS_END\nAfter\n",
        encoding="utf-8",
    )
    update_highlight_block(highlight_path, highlights)
    first_highlights = highlight_path.read_text(encoding="utf-8")
    update_highlight_block(highlight_path, highlights)
    assert highlight_path.read_text(encoding="utf-8") == first_highlights
    assert all(item in first_highlights for item in highlights)


def test_summary_provenance_rejects_tampered_csv(tmp_path):
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    final_summary = tmp_path / "final_test_summary.json"
    test_marker = tmp_path / "TEST_EVALUATED"
    plan = tmp_path / "immutable_test_plan.json"
    started = tmp_path / "TEST_EVALUATION_STARTED.json"
    final_summary.write_text('{"final": true}', encoding="utf-8")
    test_marker.write_text('{"evaluated": true}', encoding="utf-8")
    plan.write_text('{"plan": true}', encoding="utf-8")
    started.write_text('{"started": true}', encoding="utf-8")
    names = {
        "validation_long.csv": "dataset,objective\nFD001,self_scaled\n",
        "stability_long.csv": (
            "dataset,objective,cap,model_seed\n"
            "FD001,self_scaled,90,42\n"
        ),
        "exposure_bound_audit.csv": "dataset,bound_satisfied\nFD001,True\n",
            "cap_summary.csv": "dataset,objective,cap\nFD001,self_scaled,90\n",
            "aggregation_sensitivity.csv": (
                "dataset,objective,aggregation_scheme\n"
                "FD001,self_scaled,split_equal\n"
            ),
            "locked_decisions.csv": "dataset,objective\nFD001,self_scaled\n",
            "tpe_locks.csv": "dataset,objective\nFD001,self_scaled\n",
            "tpe_robustness.csv": "dataset,objective\nFD001,self_scaled\n",
        "target_support_decomposition.csv": (
            "dataset,checkpoint_objective,cap,target_scale,output_support\n"
            "FD001,self_scaled,90,candidate,candidate\n"
        ),
        "final_test.csv": "dataset,method\nFD001,fixed_cap_125\n",
        "paired_tests.csv": "dataset,comparison\nFD001,fixed_vs_self_cap_grid\n",
    }
    hashes = {}
    for name, value in names.items():
        path = summary_dir / name
        path.write_text(value, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    status = {
        "protocol": "confirmatory-summary-v2",
        "complete": True,
        "campaign_id": "campaign",
        "validation_rows": 120,
        "stability_rows": 80,
        "bound_rows": 120,
        "bound_failures": 0,
            "cap_rows": 40,
            "aggregation_sensitivity_rows": 32,
            "decision_rows": 12,
            "tpe_rows": 6,
            "tpe_robustness_rows": 6,
        "decomposition_rows": 160,
        "final_rows": 18,
        "paired_tests": 12,
        "csv_sha256": hashes,
        "final_test_summary_sha256": hashlib.sha256(
            final_summary.read_bytes()
        ).hexdigest(),
        "test_evaluated_marker_sha256": hashlib.sha256(
            test_marker.read_bytes()
        ).hexdigest(),
        "summarizer_sha256": SUMMARIZER_SHA256,
        "manuscript_asset_generator_sha256": ASSET_GENERATOR_SHA256,
    }
    (summary_dir / "summary_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    validate_summary_provenance(
        summary_dir, chain_validator=lambda *args: None
    )
    (summary_dir / "final_test.csv").write_text(
        "dataset,method\nFD001,tampered\n", encoding="utf-8"
    )
    try:
        validate_summary_provenance(
            summary_dir, chain_validator=lambda *args: None
        )
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered manuscript CSV passed provenance")


def test_summary_provenance_rejects_broken_completed_chain(tmp_path):
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    required = {
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
    hashes = {}
    for name in required:
        path = summary_dir / name
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    final_summary = tmp_path / "final_test_summary.json"
    marker = tmp_path / "TEST_EVALUATED"
    plan = tmp_path / "immutable_test_plan.json"
    started = tmp_path / "TEST_EVALUATION_STARTED.json"
    final_summary.write_text("{}", encoding="utf-8")
    marker.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    started.write_text("{}", encoding="utf-8")
    status = {
        "protocol": "confirmatory-summary-v2",
        "complete": True,
        "campaign_id": "campaign",
        "validation_rows": 120,
        "stability_rows": 80,
        "bound_rows": 120,
        "bound_failures": 0,
            "cap_rows": 40,
            "aggregation_sensitivity_rows": 32,
            "decision_rows": 12,
            "tpe_rows": 6,
            "tpe_robustness_rows": 6,
        "decomposition_rows": 160,
        "final_rows": 18,
        "paired_tests": 12,
        "csv_sha256": hashes,
        "final_test_summary_sha256": hashlib.sha256(
            final_summary.read_bytes()
        ).hexdigest(),
        "test_evaluated_marker_sha256": hashlib.sha256(
            marker.read_bytes()
        ).hexdigest(),
        "summarizer_sha256": SUMMARIZER_SHA256,
        "manuscript_asset_generator_sha256": ASSET_GENERATOR_SHA256,
    }
    (summary_dir / "summary_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )

    def reject(*args):
        raise RuntimeError("broken completed chain")

    try:
        validate_summary_provenance(
            summary_dir, chain_validator=reject
        )
    except RuntimeError as exc:
        assert "broken completed chain" in str(exc)
    else:
        raise AssertionError("Broken completed chain reached asset generation")
