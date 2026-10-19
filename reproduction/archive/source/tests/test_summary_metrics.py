import json

import numpy as np

from summarize_confirmatory import (
    add_holm_adjustment,
    aggregation_sensitivity,
    cap_summary,
    decomposition_summary,
    exact_signed_rank_test,
    nasa_score,
    paired_engine_bootstrap,
    reference_decision_metrics,
    rmse,
    validated_prediction_arrays,
    write_csv,
)


def test_cap_summary_reproduces_declared_seed_split_weighting():
    screening = []
    stability = []
    for dataset in ("FD001", "FD002", "FD003", "FD004"):
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            for cap in (90, 100, 110, 115, 125):
                base = float(cap) / 10.0
                for split_seed, value in (
                    (20260724, base),
                    (20260725, base + 3.0),
                    (20260726, base + 6.0),
                ):
                    screening.append(
                        {
                            "dataset": dataset,
                            "objective": objective,
                            "cap": cap,
                            "split_seed": split_seed,
                            "model_seed": 42,
                            metric: value,
                        }
                    )
                for seed, value in ((43, base + 1.0), (44, base + 2.0)):
                    stability.append(
                        {
                            "dataset": dataset,
                            "objective": objective,
                            "cap": cap,
                            "split_seed": 20260724,
                            "model_seed": seed,
                            metric: value,
                        }
                    )
    rows = cap_summary(screening, stability)
    row = next(
        item
        for item in rows
        if item["dataset"] == "FD001"
        and item["objective"] == "self_scaled"
        and item["cap"] == 90
    )
    # Primary mean=(9+10+11)/3=10; split values are 10,12,15.
    assert row["primary_model_seed_mean"] == 10.0
    assert row["mean_across_splits"] == 37.0 / 3.0
    assert row["selection_aggregate_complete"] is True
    sensitivity = aggregation_sensitivity(screening, stability)
    assert len(sensitivity) == 32
    assert all(item["selected_cap"] == 90 for item in sensitivity)


def test_reference_decision_metrics_use_fixed_target():
    original = np.asarray([150.0, 25.0, 10.0])
    reference = np.clip(original, 0.0, 125.0)
    prediction = np.asarray([125.0, 35.0, 15.0])
    metrics = reference_decision_metrics(prediction, reference)
    assert metrics["mean_error"] == 5.0
    assert metrics["late_prediction_fraction"] == 2.0 / 3.0
    assert metrics["delayed_action_rate_at_30"] == 0.5
    assert metrics["premature_action_rate_at_30"] == 0.0
    assert metrics["n_due_at_30"] == 2
    assert metrics["n_not_due_at_30"] == 1


def test_reference_metric_helpers_are_finite_and_exact_at_zero_error():
    target = np.asarray([125.0, 30.0, 0.0])
    assert rmse(target, target) == 0.0
    assert nasa_score(target, target) == 0.0


def test_validation_decomposition_averages_mse_not_squared_mean_rmse():
    mse_columns = (
        "candidate_target_candidate_support_mse",
        "candidate_target_common_support_mse",
        "reference_target_candidate_support_mse",
        "reference_target_common_support_mse",
    )

    def row(split, seed, value):
        output = {
            "dataset": "FD001",
            "objective": "self_scaled",
            "cap": 90,
            "split_seed": split,
            "model_seed": seed,
        }
        output.update({column: value for column in mse_columns})
        return output

    screening = [
        row(20260724, 42, 1.0),
        row(20260725, 42, 25.0),
        row(20260726, 42, 36.0),
    ]
    stability = [
        row(20260724, 43, 9.0),
        row(20260724, 44, 16.0),
    ]
    rows = decomposition_summary(
        screening, stability, allow_incomplete=True
    )
    selected = next(
        item
        for item in rows
        if item["dataset"] == "FD001"
        and item["checkpoint_objective"] == "self_scaled"
        and item["target_scale"] == "candidate"
        and item["output_support"] == "candidate"
    )
    split_values = json.loads(selected["split_values"])
    assert split_values == [26.0 / 3.0, 25.0, 36.0]
    assert selected["mean_across_splits"] == np.mean(split_values)


def test_archived_support_arrays_must_be_clips_of_the_same_raw_predictions():
    metrics = {
        "units": ["u1", "u2"],
        "raw_predictions": [140.0, -2.0],
        "candidate_support_predictions": [110.0, 0.0],
        "common_support_predictions": [125.0, 0.0],
        "original_targets": [150.0, 5.0],
    }
    units, candidate, common, target = validated_prediction_arrays(
        metrics, 110, "fixture"
    )
    assert units == ("u1", "u2")
    assert np.array_equal(candidate, [110.0, 0.0])
    assert np.array_equal(common, [125.0, 0.0])
    assert np.array_equal(target, [150.0, 5.0])

    metrics["common_support_predictions"] = [124.0, 0.0]
    try:
        validated_prediction_arrays(metrics, 110, "fixture")
    except RuntimeError as exc:
        assert "Common-support predictions" in str(exc)
    else:
        raise AssertionError("Inconsistent support array was accepted")


def test_exact_signed_rank_handles_partial_zeros_and_tied_ranks():
    statistic, pvalue, nonzero = exact_signed_rank_test(
        np.asarray([1.0, -1.0, 0.0, 2.0, -2.0])
    )
    assert statistic == 5.0
    assert pvalue == 1.0
    assert nonzero == 4

    statistic, pvalue, nonzero = exact_signed_rank_test(
        np.asarray([1.0, 0.0, 2.0, 3.0, 4.0])
    )
    assert statistic == 0.0
    assert pvalue == 0.125
    assert nonzero == 4


def test_exact_signed_rank_five_nonzero_minimum_two_sided_pvalue():
    statistic, pvalue, nonzero = exact_signed_rank_test(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    assert statistic == 0.0
    assert pvalue == 0.0625
    assert nonzero == 5


def test_holm_adjustment_is_step_down_and_order_invariant():
    rows = [
        {"wilcoxon_pvalue": 0.01},
        {"wilcoxon_pvalue": 0.04},
        {"wilcoxon_pvalue": 0.03},
    ]
    add_holm_adjustment(rows, "adjusted")
    assert [row["adjusted"] for row in rows] == [0.03, 0.06, 0.06]


def test_paired_engine_bootstrap_uses_paired_resamples_deterministically():
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    left = target + 1.0
    right = target.copy()
    first = paired_engine_bootstrap(
        left, right, target, "paired-fixture", resamples=250
    )
    second = paired_engine_bootstrap(
        left, right, target, "paired-fixture", resamples=250
    )
    assert first == second
    assert first == (1.0, 1.0, 1.0, 0.0)


def test_empty_csv_write_removes_stale_generated_table(tmp_path):
    path = tmp_path / "final_test.csv"
    path.write_text("stale\n", encoding="utf-8")
    write_csv(path, [])
    assert not path.exists()


def test_maintenance_rates_reject_empty_target_strata():
    try:
        reference_decision_metrics(
            np.asarray([10.0, 15.0]), np.asarray([10.0, 15.0])
        )
    except ValueError as exc:
        assert "empty target stratum" in str(exc)
    else:
        raise AssertionError("Empty maintenance stratum was silently scored")
