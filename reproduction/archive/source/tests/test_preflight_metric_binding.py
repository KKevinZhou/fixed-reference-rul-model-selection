from types import SimpleNamespace

import pytest

import preflight_confirmatory as preflight


def rebuilt_metrics():
    return {
        "self_scaled": {
            "own_rmse": {"mean": 8.0, "values": [8.0]},
            "reference_rmse": {"mean": 12.0, "values": [12.0]},
        },
        "fixed_reference": {
            "own_rmse": {"mean": 9.0, "values": [9.0]},
            "reference_rmse": {"mean": 10.0, "values": [10.0]},
        },
    }


def test_tpe_cross_scale_metrics_are_bound(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "validate_validation_result",
        lambda *_args, **_kwargs: rebuilt_metrics(),
    )
    trial = SimpleNamespace(
        params={
            "train_cap": 100,
            "lr": 1e-3,
            "batch": 128,
            "huber_delta": 1.0,
        },
        user_attrs={"own_rmse": 8.0, "reference_rmse": 10.0},
        value=10.0,
    )
    result = {
        "config": {
            "seq_len": 50,
            "lr": 1e-3,
            "batch": 128,
            "huber_delta": 1.0,
        }
    }
    manifest = {
        "objective": "fixed_reference",
        "campaign_id": "campaign",
        "dataset": "FD001",
        "split_seed": 20260724,
        "model_seed": 42,
        "fixed_seq_len": 50,
    }
    assert preflight.validate_tpe_result(
        "result.json", result, manifest, {}, trial
    ) == 10.0
    trial.user_attrs["own_rmse"] = 99.0
    with pytest.raises(RuntimeError, match="own_rmse"):
        preflight.validate_tpe_result(
            "result.json", result, manifest, {}, trial
        )


def test_robustness_metric_summary_is_bound():
    entry = {
        "split_seed": 20260724,
        "model_seeds": [43, 44],
        "selection_metric": "reference_rmse",
        "values": [10.0, 11.0],
    }
    preflight.verify_metric_summary_entry(
        entry,
        split_seed=20260724,
        model_seeds=[43, 44],
        selection_metric="reference_rmse",
        expected_values=[10.0, 11.0],
        label="FD001_fixed_reference",
    )
    entry["values"][1] = 99.0
    with pytest.raises(RuntimeError, match="archived predictions"):
        preflight.verify_metric_summary_entry(
            entry,
            split_seed=20260724,
            model_seeds=[43, 44],
            selection_metric="reference_rmse",
            expected_values=[10.0, 11.0],
            label="FD001_fixed_reference",
        )


def test_split_choice_comparison_tolerates_roundoff():
    observed = [
        {
            "split_seed": 20260724,
            "cap": 90,
            "objective_value": 8.691498738510738,
        }
    ]
    rebuilt = [
        {
            "split_seed": 20260724,
            "cap": 90,
            "objective_value": 8.69149873851074,
        }
    ]
    preflight.verify_split_choices(observed, rebuilt, "FD001 self")
    rebuilt[0]["cap"] = 100
    with pytest.raises(RuntimeError, match="Split choice"):
        preflight.verify_split_choices(observed, rebuilt, "FD001 self")
