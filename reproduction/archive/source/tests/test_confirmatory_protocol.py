import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confirmatory_rul_train import (
    metric_bundle,
    split_units,
    validation_endpoints,
)
from preflight_confirmatory import campaign_result_path, recompute_validation_metrics
from run_confirmatory_final import carrier_transfer_gate_passes


def toy_frame(units=20, lifetime=160):
    rows = []
    for unit in range(1, units + 1):
        unit_life = lifetime + unit
        for cycle in range(1, unit_life + 1):
            rows.append({"unit": unit, "cycle": cycle})
    frame = pd.DataFrame(rows)
    maximum = frame.groupby("unit")["cycle"].transform("max")
    frame["RUL_ORIGINAL"] = maximum - frame["cycle"]
    return frame


def test_preflight_rejects_manifest_paths_outside_campaign_results(tmp_path):
    root = tmp_path / "campaign"
    results = root / "results"
    results.mkdir(parents=True)
    inside = results / "FD001_validation.json"
    inside.write_text("{}", encoding="utf-8")
    assert campaign_result_path(root, str(inside), "fixture") == inside.resolve()

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        campaign_result_path(root, str(outside), "fixture")
    except RuntimeError as exc:
        assert "escapes the campaign results directory" in str(exc)
    else:
        raise AssertionError("Outside result path passed preflight boundary")


def test_group_split_is_deterministic_and_disjoint():
    frame = toy_frame()
    train_a, val_a = split_units(frame, 20260724, 0.15)
    train_b, val_b = split_units(frame, 20260724, 0.15)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert not set(train_a).intersection(val_a)
    assert set(train_a).union(val_a) == set(frame["unit"].unique())


def test_final_runner_accepts_only_frozen_pretest_carrier_outcomes():
    base = {"selection_frozen": True, "test_evaluated": False}
    assert carrier_transfer_gate_passes({**base, "status": "complete"})
    assert carrier_transfer_gate_passes(
        {**base, "status": "terminal_failure"}
    )
    assert not carrier_transfer_gate_passes(
        {**base, "status": "in_progress"}
    )
    assert not carrier_transfer_gate_passes(
        {
            "status": "terminal_failure",
            "selection_frozen": False,
            "test_evaluated": False,
        }
    )
    assert not carrier_transfer_gate_passes(
        {
            "status": "complete",
            "selection_frozen": True,
            "test_evaluated": True,
        }
    )


def test_endpoint_schedule_is_one_per_unit_and_prefix_only():
    frame = toy_frame()
    _, validation = split_units(frame, 20260724, 0.15)
    prefix, targets, schedule = validation_endpoints(
        frame, validation, 50, 125, 21260727
    )
    assert len(schedule) == len(validation)
    assert set(targets) == set(validation)
    for row in schedule:
        observed = prefix.loc[
            prefix["unit"] == row["unit"], "cycle"
        ].max()
        assert observed == row["observed_cycles"]
        assert row["original_rul"] == row["lifetime"] - observed


def test_bound_uses_candidate_support_not_common_support():
    target = {1: 120.0, 2: 110.0, 3: 40.0}
    raw = {1: 120.0, 2: 110.0, 3: 40.0}
    bundle = metric_bundle(raw, target, train_cap=90, reporting_cap=125)
    candidate_original = np.asarray(
        bundle["candidate_support_predictions"]
    )
    original = np.asarray(bundle["original_targets"])
    hidden_gap = (
        np.mean((candidate_original - original) ** 2)
        - bundle["own_rmse"] ** 2
    )
    assert hidden_gap + 1e-10 >= bundle["exposure_bound"] ** 2
    reference = np.clip(original, 0.0, 125.0)
    reference_gap = (
        np.mean((candidate_original - reference) ** 2)
        - bundle["own_rmse"] ** 2
    )
    reference_exposure = np.mean(
        np.maximum(reference - 90.0, 0.0) ** 2
    )
    assert reference_gap + 1e-10 >= reference_exposure
    assert bundle["reference_rmse"] == 0.0
    assert bundle["original_rmse"] == 0.0


def test_reference_support_is_common_across_caps():
    target = {1: 120.0, 2: 80.0}
    raw = {1: 110.0, 2: 70.0}
    low = metric_bundle(raw, target, train_cap=90, reporting_cap=125)
    high = metric_bundle(raw, target, train_cap=115, reporting_cap=125)
    assert low["common_support_predictions"] == high[
        "common_support_predictions"
    ]
    assert low["reference_rmse"] == high["reference_rmse"]
    assert low["candidate_support_predictions"] != high[
        "candidate_support_predictions"
    ]


def test_preflight_recomputes_validation_objectives_from_predictions():
    own = math.sqrt(0.5)
    reference = math.sqrt(5.0)
    metrics = {
        "units": [1, 2],
        "raw_predictions": [8.0, 15.0],
        "candidate_support_predictions": [8.0, 10.0],
        "common_support_predictions": [8.0, 15.0],
        "original_targets": [9.0, 18.0],
        "own_rmse": own,
        "reference_rmse": reference,
        "original_rmse": reference,
    }
    result = {
        "train_cap": 10,
        "reporting_cap": 20,
        "model_seeds": [42],
        "validation_units": [1, 2],
        "validation_endpoints": [
            {"unit": 1, "original_rul": 9.0},
            {"unit": 2, "original_rul": 18.0},
        ],
        "per_seed_validation": [
            {
                "seed": 42,
                "checkpoints": {
                    objective: {"metrics": dict(metrics)}
                    for objective in (
                        "self_scaled",
                        "fixed_reference",
                    )
                },
            }
        ],
        "validation_summary": {
            objective: {
                "own_rmse": {
                    "values": [own],
                    "mean": own,
                    "std": 0.0,
                },
                "reference_rmse": {
                    "values": [reference],
                    "mean": reference,
                    "std": 0.0,
                },
                "original_rmse": {
                    "values": [reference],
                    "mean": reference,
                    "std": 0.0,
                },
            }
            for objective in ("self_scaled", "fixed_reference")
        },
    }
    rebuilt = recompute_validation_metrics(result, "fixture.json")
    assert math.isclose(
        rebuilt["self_scaled"][
            "candidate_target_common_support_rmse"
        ]["mean"],
        math.sqrt(13.0),
    )
    result["validation_summary"]["self_scaled"]["own_rmse"][
        "mean"
    ] = 0.0
    try:
        recompute_validation_metrics(result, "fixture.json")
    except RuntimeError as exc:
        assert "differs from predictions" in str(exc)
    else:
        raise AssertionError("Tampered validation scalar passed preflight")


def test_test_reader_is_guarded_by_validation_only_branch():
    """Regression guard: validation mode cannot reach the test parser."""
    tree = ast.parse(
        (ROOT / "confirmatory_rul_train.py").read_text(encoding="utf-8")
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    def reader_calls(nodes):
        return [
            node
            for statement in nodes
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "read_raw_test"
        ]

    guards = [
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Attribute)
        and isinstance(node.test.operand.value, ast.Name)
        and node.test.operand.value.id == "args"
        and node.test.operand.attr == "validation_only"
    ]
    assert len(guards) == 1
    guard = guards[0]
    assert len(reader_calls(guard.body)) == 1
    assert reader_calls(guard.orelse) == []
    all_calls = reader_calls(main.body)
    assert len(all_calls) == 1
