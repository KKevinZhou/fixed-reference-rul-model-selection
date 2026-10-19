import json
import sys

import pytest

import preflight_confirmatory as preflight
import retry_carrier_transfer_failures as retry


def test_discover_retry_is_limited_to_logged_nonfinite_cell(
    tmp_path, monkeypatch
):
    transfer = tmp_path / "carrier_transfer"
    logs = transfer / "logs"
    logs.mkdir(parents=True)
    plan = {
        "campaign_id": "carrierv1-test",
        "epochs": 40,
        "patience": 6,
        "train_stride": 5,
        "validation_interval": 2,
    }
    source = {}
    monkeypatch.setattr(retry.carrier, "DATASETS", ("FD002",))
    monkeypatch.setattr(retry.carrier, "CAPS", (110,))
    monkeypatch.setattr(retry.carrier, "valid_result", lambda *_a, **_k: False)
    log = logs / "FD002_cap110_carrierv1-test.log"
    log.write_text(
        "Traceback\nFloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )

    cells = retry.discover_failed_cells(
        data_root=tmp_path / "data",
        transfer_root=transfer,
        plan=plan,
        source_manifest=source,
    )

    assert [(cell["dataset"], cell["cap"]) for cell in cells] == [
        ("FD002", 110)
    ]
    assert cells[0]["failure_signature"] == retry.FAILURE_SIGNATURE
    assert str((tmp_path / "data").resolve()) in cells[0]["command_tail"]


def test_discover_retry_rejects_unclassified_failure(tmp_path, monkeypatch):
    transfer = tmp_path / "carrier_transfer"
    logs = transfer / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(retry.carrier, "DATASETS", ("FD002",))
    monkeypatch.setattr(retry.carrier, "CAPS", (115,))
    monkeypatch.setattr(retry.carrier, "valid_result", lambda *_a, **_k: False)
    (logs / "FD002_cap115_carrierv1-test.log").write_text(
        "Process stopped for an unrelated reason\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="only for an explicitly recorded"):
        retry.discover_failed_cells(
            data_root=tmp_path / "data",
            transfer_root=transfer,
            plan={"campaign_id": "carrierv1-test"},
            source_manifest={},
        )


def test_retry_refuses_any_standard_test_state(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "TEST_EVALUATION_STARTED.json").write_text(
        json.dumps({"started": True}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retry_carrier_transfer_failures.py",
            "--data_root",
            str(tmp_path / "data"),
            "--campaign_root",
            str(campaign),
        ],
    )
    with pytest.raises(RuntimeError, match="forbidden after standard-test"):
        retry.main()


def test_preflight_binds_complete_exact_retry(tmp_path):
    campaign_id = "carrier-test"
    transfer = tmp_path / "carrier_transfer"
    logs = transfer / "logs"
    results = transfer / "results"
    logs.mkdir(parents=True)
    results.mkdir()
    plan = {
        "campaign_id": campaign_id,
        "datasets": ["FD002"],
        "caps": [110],
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
    }
    plan_path = tmp_path / "carrier_transfer_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    original = logs / f"FD002_cap110_{campaign_id}.log"
    original.write_text(
        "FloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )
    retry_log = logs / f"FD002_cap110_{campaign_id}_retry1.log"
    retry_log.write_text("completed\n", encoding="utf-8")
    result = results / "FD002_cap110.json"
    result.write_text('{"status": "complete"}\n', encoding="utf-8")
    manifest = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": campaign_id,
        "carrier_plan_sha256": preflight.sha256_file(plan_path),
        "retry_helper_sha256": preflight.sha256_file(
            retry.HERE / "retry_carrier_transfer_failures.py"
        ),
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "attempt_limit_per_cell": 1,
        "execution_mode": "sequential",
        "parameter_changes": False,
        "test_files_permitted": False,
        "cells": [
            {
                "dataset": "FD002",
                "cap": 110,
                "original_log": str(original),
                "original_log_sha256": preflight.sha256_file(original),
                "failure_signature": retry.FAILURE_SIGNATURE,
            }
        ],
    }
    manifest_path = tmp_path / "carrier_transfer_retry_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "protocol": manifest["protocol"],
        "campaign_id": campaign_id,
        "manifest_sha256": preflight.sha256_file(manifest_path),
        "status": "complete",
        "test_evaluated": False,
        "outcomes": [
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 0,
                "result_valid": True,
                "retry_log": str(retry_log),
                "retry_log_sha256": preflight.sha256_file(retry_log),
                "result_path": str(result),
                "result_sha256": preflight.sha256_file(result),
            }
        ],
    }
    (tmp_path / "carrier_transfer_retry_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    carrier_summary = {
        "result_sha256": {
            str(result.resolve()): preflight.sha256_file(result)
        }
    }

    files, failed = preflight.carrier_retry_audit(
        tmp_path, plan, carrier_summary
    )

    assert failed == set()
    assert len(files) == 4
    assert str(original.resolve()) in files
    assert str(retry_log.resolve()) in files


def test_preflight_binds_terminal_exact_retry_failure(tmp_path):
    campaign_id = "carrier-test"
    transfer = tmp_path / "carrier_transfer"
    logs = transfer / "logs"
    logs.mkdir(parents=True)
    plan = {
        "campaign_id": campaign_id,
        "datasets": ["FD002"],
        "caps": [110],
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
    }
    plan_path = tmp_path / "carrier_transfer_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    original = logs / f"FD002_cap110_{campaign_id}.log"
    original.write_text(
        "FloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )
    retry_log = logs / f"FD002_cap110_{campaign_id}_retry1.log"
    retry_log.write_text(
        "FloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )
    result = transfer / "results" / "FD002_cap110.json"
    manifest = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": campaign_id,
        "carrier_plan_sha256": preflight.sha256_file(plan_path),
        "retry_helper_sha256": preflight.sha256_file(
            retry.HERE / "retry_carrier_transfer_failures.py"
        ),
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "attempt_limit_per_cell": 1,
        "execution_mode": "sequential",
        "parameter_changes": False,
        "test_files_permitted": False,
        "cells": [
            {
                "dataset": "FD002",
                "cap": 110,
                "original_log": str(original),
                "original_log_sha256": preflight.sha256_file(original),
                "failure_signature": retry.FAILURE_SIGNATURE,
            }
        ],
    }
    manifest_path = tmp_path / "carrier_transfer_retry_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    retry_summary = {
        "protocol": manifest["protocol"],
        "campaign_id": campaign_id,
        "manifest_sha256": preflight.sha256_file(manifest_path),
        "status": "failed",
        "test_evaluated": False,
        "outcomes": [
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 1,
                "result_valid": False,
                "failure_signature_recurred": True,
                "retry_log": str(retry_log),
                "retry_log_sha256": preflight.sha256_file(retry_log),
                "result_path": str(result),
            }
        ],
    }
    retry_summary_path = (
        tmp_path / "carrier_transfer_retry_summary.json"
    )
    retry_summary_path.write_text(
        json.dumps(retry_summary), encoding="utf-8"
    )
    carrier_summary = {
        "status": "terminal_failure",
        "missing_cells": [{"dataset": "FD002", "cap": 110}],
        "failed_retry_outcomes": [
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 1,
                "result_valid": False,
                "failure_signature_recurred": True,
            }
        ],
        "retry_summary_sha256": preflight.sha256_file(
            retry_summary_path
        ),
        "result_sha256": {},
    }

    files, failed = preflight.carrier_retry_audit(
        tmp_path, plan, carrier_summary
    )

    assert failed == {("FD002", 110)}
    assert len(files) == 4


def test_terminal_summary_suppresses_incomplete_subset_selection(
    tmp_path, monkeypatch
):
    campaign = tmp_path / "campaign"
    transfer = campaign / "carrier_transfer"
    results = transfer / "results"
    results.mkdir(parents=True)
    plan = {
        "protocol": "confirmatory-carrier-transfer-v1",
        "campaign_id": "carrier-test",
    }
    (campaign / "carrier_transfer_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    retry_summary = campaign / "carrier_transfer_retry_summary.json"
    retry_summary.write_text('{"status":"failed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        retry.carrier, "DATASETS", ("FD001", "FD002")
    )
    monkeypatch.setattr(retry.carrier, "CAPS", (90, 110))

    for dataset, cap, own, fixed in (
        ("FD001", 90, 7.0, 12.0),
        ("FD001", 110, 10.0, 9.0),
        ("FD002", 90, 8.0, 11.0),
    ):
        path = retry.result_path(transfer, dataset, cap)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "validation_summary": {
                        "self_scaled": {
                            "own_rmse": {"mean": own},
                            "reference_rmse": {"mean": own + 1.0},
                        },
                        "fixed_reference": {
                            "reference_rmse": {"mean": fixed}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        retry.carrier,
        "valid_result",
        lambda path, **_kwargs: path.is_file(),
    )

    terminal = retry.terminal_carrier_summary(
        campaign_root=campaign,
        transfer_root=transfer,
        plan=plan,
        source_manifest={},
        retry_summary_path=retry_summary,
        outcomes=[
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 1,
                "result_valid": False,
                "failure_signature_recurred": True,
                "retry_log": "retry.log",
                "retry_log_sha256": "a" * 64,
            }
        ],
    )

    assert terminal["status"] == "terminal_failure"
    assert set(terminal["selections"]) == {"FD001"}
    assert terminal["missing_cells"] == [
        {"dataset": "FD002", "cap": 110}
    ]
    assert len(terminal["result_sha256"]) == 3
