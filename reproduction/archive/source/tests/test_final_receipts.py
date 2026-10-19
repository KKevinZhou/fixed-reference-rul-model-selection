import ast
import json
from pathlib import Path
from types import SimpleNamespace

import final_attested_trainer as attested
import pytest
import run_confirmatory_final as final_runner
from run_confirmatory_final import (
    acquire_single_instance_lock,
    attempt_integrity_payload,
    config_matches,
    ensure_launch_authorization,
    execution_environment,
    expected_config,
    planned_result_paths,
    per_seed_prediction_payload_valid,
    receipt_integrity_payload,
    record_attempt_completion,
    record_evaluation_attempt,
    recover_receipt,
    release_single_instance_lock,
    result_integrity_payload,
    sha256_file,
    stable_hash,
    validate_completed_chain,
    validate_final,
    validate_result_receipt,
    validate_summary_chain,
    write_result_receipt,
)

RUNTIME_ENVIRONMENT = final_runner.capture_runtime_environment()


def manifest():
    return {
        "campaign_id": "screening-test",
        "epochs": 40,
        "patience": 6,
        "train_stride": 5,
        "validation_interval": 2,
        "carrier": "compact_tcn",
        "trainer_sha256": "trainer",
        "model_sha256": "model",
        "raw_train_sha256": {"FD001": "train"},
    }


def canonical():
    payload = {
        "units": [1, 2],
        "original_targets": [9.0, 130.0],
    }
    return {
        "test_filename": "test_FD001.txt",
        "truth_filename": "RUL_FD001.txt",
        "test_sha256": "test",
        "truth_sha256": "truth",
        "test_unit_count": 2,
        "ordered_unit_target_sha256": stable_hash(payload),
    }


def bind_official_boundary(output, plan, started):
    plan.update(
        {
            "campaign_id": "screening-test",
            "canonical_source": {
                "canonical_manifest_sha256": "c" * 64,
            },
            "trainer_sha256": sha256_file(final_runner.TRAINER),
            "model_sha256": sha256_file(final_runner.MODEL_SOURCE),
            "final_runner_sha256": sha256_file(
                Path(final_runner.__file__).resolve()
            ),
            "final_attested_trainer_sha256": sha256_file(
                final_runner.FINAL_ATTESTED_TRAINER
            ),
            "summarizer_sha256": sha256_file(
                final_runner.SUMMARIZER
            ),
            "manuscript_asset_generator_sha256": sha256_file(
                final_runner.MANUSCRIPT_ASSET_GENERATOR
            ),
            "environment_lock_sha256": sha256_file(
                final_runner.ENVIRONMENT_LOCK
            ),
            "runtime_environment": dict(RUNTIME_ENVIRONMENT),
        }
    )
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text(
        json.dumps(
            {
                "campaign_id": "screening-test",
                "test_plan_sha256": stable_hash(plan),
                "selection_changes_forbidden": True,
            }
        ),
        encoding="utf-8",
    )
    verified_files = {
        dataset: {
            "test_filename": value["test_filename"],
            "test_sha256": value["test_sha256"],
            "truth_filename": value["truth_filename"],
            "truth_sha256": value["truth_sha256"],
        }
        for dataset, value in plan["canonical_test_data"].items()
    }
    receipt = {
        "protocol": "confirmatory-local-test-data-verification-v2",
        "campaign_id": "screening-test",
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": sha256_file(started),
        "canonical_manifest_sha256": "c" * 64,
        "verified_files": verified_files,
        "verified_unix": 1.0,
    }
    path = output / "TEST_DATA_VERIFIED.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_planned_execution_source_hashes_include_model_source():
    observed = final_runner.planned_execution_source_hashes(
        {
            "trainer_sha256": "trainer",
            "model_sha256": "model",
        }
    )
    assert observed["trainer_sha256"] == "trainer"
    assert observed["model_sha256"] == "model"
    assert observed["final_runner_sha256"] == sha256_file(
        Path(final_runner.__file__).resolve()
    )


def objective_result(offset):
    bundles = []
    rows = []
    for seed_index, seed in enumerate([45, 46, 47, 48, 49]):
        raw = {
            1: 10.0 + offset + seed_index,
            2: 128.0 + offset - seed_index,
        }
        bundle = attested.frozen.metric_bundle(
            raw,
            {1: 9.0, 2: 130.0},
            125.0,
            125.0,
        )
        bundle.update(
            attested.frozen.decision_metrics(
                bundle["common_support_predictions"], [9.0, 125.0]
            )
        )
        bundles.append(bundle)
        rows.append({"seed": seed, **bundle})
    scalar_names = attested.frozen.scalar_metrics(bundles[0])
    raw_ensemble = [
        sorted(
            bundle["raw_predictions"][index]
            for bundle in bundles
        )[2]
        for index in range(2)
    ]
    common_ensemble = [
        sorted(
            bundle["common_support_predictions"][index]
            for bundle in bundles
        )[2]
        for index in range(2)
    ]
    candidate_ensemble = [
        sorted(
            bundle["candidate_support_predictions"][index]
            for bundle in bundles
        )[2]
        for index in range(2)
    ]
    ensemble = attested.frozen.metric_bundle(
        {1: raw_ensemble[0], 2: raw_ensemble[1]},
        {1: 9.0, 2: 130.0},
        125.0,
        125.0,
    )
    ensemble["common_support_predictions"] = common_ensemble
    ensemble["candidate_support_predictions"] = candidate_ensemble
    ensemble.update(
        attested.frozen.decision_metrics(
            ensemble["common_support_predictions"], [9.0, 125.0]
        )
    )
    return {
        "per_seed": rows,
        "summary": {
            key: attested.frozen.aggregate(
                [bundle[key] for bundle in bundles]
            )
            for key in scalar_names
        },
        "ensemble": ensemble,
    }


def official_result():
    return {
        "protocol": "confirmatory-rul-cap-v2",
        "campaign_id": "screening-test",
        "dataset": "FD001",
        "tag": "locked",
        "validation_only": False,
        "test_evaluated": True,
        "train_cap": 125,
        "reporting_cap": 125.0,
        "split_seed": 20260724,
        "endpoint_seed": 21260727,
        "model_seeds": [45, 46, 47, 48, 49],
        "config": expected_config(manifest(), "FD001"),
        "source_fingerprints": {
            "trainer_sha256": "trainer",
            "model_sha256": "model",
        },
        "data_fingerprints": {
            "train_sha256": "train",
            "test_sha256": "test",
            "truth_sha256": "truth",
        },
        "test_unit_count": 2,
        "environment": dict(RUNTIME_ENVIRONMENT),
        "test_results": {
            "self_scaled": objective_result(0.0),
            "fixed_reference": objective_result(1.0),
        },
    }


def official_spec(filename="FD001_locked.json"):
    return {
        "result_filename": filename,
        "campaign_id": "screening-test",
        "dataset": "FD001",
        "tag": "locked",
        "train_cap": 125,
        "reporting_cap": 125.0,
        "split_seed": 20260724,
        "endpoint_seed": 21260727,
        "model_seeds": [45, 46, 47, 48, 49],
        "config": expected_config(manifest(), "FD001"),
        "trainer_sha256": "trainer",
        "model_sha256": "model",
        "train_sha256": "train",
        "runtime_environment": dict(RUNTIME_ENVIRONMENT),
    }


@pytest.mark.parametrize("changed_source", ["trainer", "model"])
def test_launch_rejects_changed_frozen_source_before_authorization(
    tmp_path, monkeypatch, changed_source
):
    trainer = tmp_path / "confirmatory_rul_train.py"
    model = tmp_path / "models_v3.py"
    trainer.write_bytes(final_runner.TRAINER.read_bytes())
    model.write_bytes(final_runner.MODEL_SOURCE.read_bytes())
    monkeypatch.setattr(final_runner, "TRAINER", trainer)
    monkeypatch.setattr(final_runner, "MODEL_SOURCE", model)

    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    result_path.parent.mkdir(parents=True)
    started = output / "TEST_EVALUATION_STARTED.json"
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)

    changed_path = trainer if changed_source == "trainer" else model
    changed_path.write_bytes(changed_path.read_bytes() + b"\n# changed\n")
    with pytest.raises(RuntimeError, match="source changed"):
        ensure_launch_authorization(
            output, result_path, spec, plan, started
        )
    launch_path, _ = final_runner.receipt_paths(output, result_path)
    assert not launch_path.exists()


def record_successful_attempt(output, result_path, spec, plan, started):
    log_path = output / "final_logs" / f"{result_path.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_path, complete_path = record_evaluation_attempt(
        output,
        result_path,
        spec,
        plan,
        started,
        ["python", "final_attested_trainer.py"],
        log_path,
    )
    log_path.write_text("completed", encoding="utf-8")
    record_attempt_completion(
        start_path, complete_path, 0, log_path
    )


def test_full_config_validator_rejects_changed_ema():
    expected = expected_config(manifest(), "FD001")
    assert config_matches(dict(expected), expected)
    changed = dict(expected)
    changed["ema_span"] = 4
    assert not config_matches(changed, expected)


def test_final_validator_binds_endpoint_config_and_test_hash(tmp_path):
    path = tmp_path / "result.json"
    result = official_result()
    path.write_text(json.dumps(result), encoding="utf-8")
    args = SimpleNamespace(
        primary_split_seed=20260724,
        seed_base=45,
        final_seeds=5,
    )
    canonical_data = canonical()
    assert validate_final(
        path, manifest(), "FD001", 125, args, canonical_data
    )
    result["endpoint_seed"] += 1
    path.write_text(json.dumps(result), encoding="utf-8")
    assert not validate_final(
        path, manifest(), "FD001", 125, args, canonical_data
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "unit",
        "test_count",
        "candidate_clipping",
        "missing_summary",
        "summary_mean",
        "missing_ensemble",
        "ensemble_prediction",
    ],
)
def test_prediction_payload_rejects_semantic_tampering(mutation):
    result = json.loads(json.dumps(official_result()))
    objective = result["test_results"]["self_scaled"]
    if mutation == "target":
        objective["per_seed"][0]["original_targets"][0] = 8.0
    elif mutation == "unit":
        objective["per_seed"][0]["units"][1] = 3
    elif mutation == "test_count":
        result["test_unit_count"] = 1
    elif mutation == "candidate_clipping":
        objective["per_seed"][0]["candidate_support_predictions"][0] += 1.0
    elif mutation == "missing_summary":
        del objective["summary"]
    elif mutation == "summary_mean":
        objective["summary"]["reference_rmse"]["mean"] += 1.0
    elif mutation == "missing_ensemble":
        del objective["ensemble"]
    elif mutation == "ensemble_prediction":
        objective["ensemble"]["common_support_predictions"][0] += 1.0
    assert not per_seed_prediction_payload_valid(result, canonical())


def test_receipt_binds_plan_marker_result_and_fingerprints(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    result_path.parent.mkdir(parents=True)
    started = output / "TEST_EVALUATION_STARTED.json"
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    environment = execution_environment(
        output, result_path, spec, plan, started
    )
    record_successful_attempt(
        output, result_path, spec, plan, started
    )
    result = official_result()
    result["official_execution_attestation"] = {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": environment["CCV2_TEST_PLAN_SHA256"],
        "start_marker_sha256": environment[
            "CCV2_START_MARKER_SHA256"
        ],
        "launch_authorization_sha256": environment[
            "CCV2_LAUNCH_SHA256"
        ],
        "launch_nonce": environment["CCV2_LAUNCH_NONCE"],
    }
    result_path.write_text(
        json.dumps(result), encoding="utf-8"
    )
    write_result_receipt(output, result_path, spec, plan, started)
    assert validate_result_receipt(
        output, result_path, spec, plan, started
    )

    tampered = dict(result)
    tampered["train_cap"] = 115
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert not validate_result_receipt(
        output, result_path, spec, plan, started
    )


def test_receipt_rejects_runtime_environment_drift(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    result_path.parent.mkdir(parents=True)
    started = output / "TEST_EVALUATION_STARTED.json"
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    environment = execution_environment(
        output, result_path, spec, plan, started
    )
    record_successful_attempt(
        output, result_path, spec, plan, started
    )
    result = official_result()
    result["official_execution_attestation"] = {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": environment["CCV2_TEST_PLAN_SHA256"],
        "start_marker_sha256": environment[
            "CCV2_START_MARKER_SHA256"
        ],
        "launch_authorization_sha256": environment[
            "CCV2_LAUNCH_SHA256"
        ],
        "launch_nonce": environment["CCV2_LAUNCH_NONCE"],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    write_result_receipt(output, result_path, spec, plan, started)
    assert validate_result_receipt(
        output, result_path, spec, plan, started
    )
    result["environment"]["torch"] = "changed"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not validate_result_receipt(
        output, result_path, spec, plan, started
    )


def test_valid_orphaned_atomic_result_recovers_attempt_and_receipt(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    result_path.parent.mkdir(parents=True)
    started = output / "TEST_EVALUATION_STARTED.json"
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    environment = execution_environment(
        output, result_path, spec, plan, started
    )
    log_path = output / "final_logs" / "orphan.log"
    log_path.parent.mkdir(parents=True)
    start_path, complete_path = record_evaluation_attempt(
        output,
        result_path,
        spec,
        plan,
        started,
        ["python", "final_attested_trainer.py"],
        log_path,
    )
    log_path.write_text("atomic result written", encoding="utf-8")
    result = official_result()
    result["official_execution_attestation"] = {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": environment["CCV2_TEST_PLAN_SHA256"],
        "start_marker_sha256": environment[
            "CCV2_START_MARKER_SHA256"
        ],
        "launch_authorization_sha256": environment[
            "CCV2_LAUNCH_SHA256"
        ],
        "launch_nonce": environment["CCV2_LAUNCH_NONCE"],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not complete_path.exists()
    assert recover_receipt(
        output,
        result_path,
        spec,
        plan,
        started,
        result_is_valid=True,
    )
    recovered = json.loads(complete_path.read_text(encoding="utf-8"))
    assert (
        recovered["protocol"]
        == "confirmatory-final-attempt-recovered-v2"
    )
    assert recovered["result_sha256"] == sha256_file(result_path)
    assert recovered["attempt_start_sha256"] == sha256_file(start_path)
    assert validate_result_receipt(
        output, result_path, spec, plan, started
    )


def test_attempt_ledger_retains_interrupted_and_completed_attempts(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    started = output / "TEST_EVALUATION_STARTED.json"
    started.parent.mkdir(parents=True)
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    first_log = output / "final_logs" / "first.log"
    first_log.parent.mkdir(parents=True)
    first_start, first_complete = record_evaluation_attempt(
        output,
        result_path,
        spec,
        plan,
        started,
        ["python", "trainer.py"],
        first_log,
    )
    assert first_start.is_file()
    assert not first_complete.exists()

    second_log = output / "final_logs" / "second.log"
    second_start, second_complete = record_evaluation_attempt(
        output,
        result_path,
        spec,
        plan,
        started,
        ["python", "trainer.py"],
        second_log,
    )
    second_log.write_text("completed", encoding="utf-8")
    record_attempt_completion(
        second_start, second_complete, 0, second_log
    )
    ledger = attempt_integrity_payload(output, result_path)
    assert len(ledger) == 3
    assert str(first_start.resolve()) in ledger
    assert str(second_start.resolve()) in ledger
    assert str(second_complete.resolve()) in ledger


def test_attempt_ledger_rejects_third_attempt(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    started = output / "TEST_EVALUATION_STARTED.json"
    started.parent.mkdir(parents=True)
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    for name in ("first.log", "second.log"):
        record_evaluation_attempt(
            output,
            result_path,
            spec,
            plan,
            started,
            ["python", "trainer.py"],
            output / "final_logs" / name,
        )
    with pytest.raises(RuntimeError, match="attempt limit"):
        record_evaluation_attempt(
            output,
            result_path,
            spec,
            plan,
            started,
            ["python", "trainer.py"],
            output / "final_logs" / "third.log",
        )


def test_completed_nonzero_attempt_is_terminal(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    started = output / "TEST_EVALUATION_STARTED.json"
    started.parent.mkdir(parents=True)
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    log = output / "final_logs" / "failed.log"
    log.parent.mkdir(parents=True)
    start, complete = record_evaluation_attempt(
        output,
        result_path,
        spec,
        plan,
        started,
        ["python", "trainer.py"],
        log,
    )
    log.write_text("failed", encoding="utf-8")
    record_attempt_completion(start, complete, 1, log)
    with pytest.raises(RuntimeError, match="terminal"):
        record_evaluation_attempt(
            output,
            result_path,
            spec,
            plan,
            started,
            ["python", "trainer.py"],
            output / "final_logs" / "retry.log",
        )


def test_attestation_shim_writes_launch_chain(monkeypatch, tmp_path):
    values = {
        "CCV2_TEST_PLAN_SHA256": "1" * 64,
        "CCV2_START_MARKER_SHA256": "2" * 64,
        "CCV2_LAUNCH_SHA256": "3" * 64,
        "CCV2_LAUNCH_NONCE": "4" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        attested.sys, "argv", ["final_attested_trainer.py"]
    )
    path = tmp_path / "attested.json"
    payload = {
        "protocol": "confirmatory-rul-cap-v2",
        "validation_only": False,
        "test_evaluated": True,
    }
    original_atomic = attested.frozen.atomic_json

    def fake_evaluate(*_args, **_kwargs):
        output = {}
        raw_by_objective = {
            "self_scaled": {1: 80.0, 2: 90.0},
            "fixed_reference": {1: 70.0, 2: 80.0},
        }
        for objective in ("self_scaled", "fixed_reference"):
            seed_bundle = attested.frozen.metric_bundle(
                raw_by_objective[objective],
                {1: 85.0, 2: 130.0},
                100.0,
                125.0,
            )
            ensemble_bundle = attested.frozen.metric_bundle(
                raw_by_objective[objective],
                {1: 85.0, 2: 130.0},
                100.0,
                125.0,
            )
            output[objective] = {
                "per_seed": [
                    {
                        "seed": 42,
                        **attested.frozen.scalar_metrics(seed_bundle),
                    }
                ],
                "summary": {},
                "ensemble": ensemble_bundle,
            }
        return output

    def fake_main():
        payload["test_results"] = attested.frozen.evaluate_test(
            SimpleNamespace(train_cap=100.0, reporting_cap=125.0)
        )
        attested.frozen.atomic_json(path, payload)

    monkeypatch.setattr(attested.frozen, "evaluate_test", fake_evaluate)
    monkeypatch.setattr(attested.frozen, "main", fake_main)
    attested.main()
    observed = json.loads(path.read_text(encoding="utf-8"))
    assert observed["official_execution_attestation"] == {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": "1" * 64,
        "start_marker_sha256": "2" * 64,
        "launch_authorization_sha256": "3" * 64,
        "launch_nonce": "4" * 64,
    }
    per_seed = observed["test_results"]["fixed_reference"]["per_seed"][0]
    assert per_seed["units"] == [1, 2]
    assert per_seed["raw_predictions"] == [70.0, 80.0]
    assert per_seed["mean_error"] == pytest.approx(-30.0)
    assert attested.frozen.atomic_json is original_atomic


def test_single_instance_lock_rejects_second_acquisition(tmp_path):
    lock_path = tmp_path / "official.lock"
    acquire_single_instance_lock(lock_path)
    try:
        try:
            acquire_single_instance_lock(lock_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Second process lock was not rejected")
    finally:
        release_single_instance_lock()


def test_planned_result_paths_reject_duplicate_names(tmp_path):
    plan = {
        "cap_result_specs": {
            "FD001": {"125": {"result_filename": "same.json"}}
        },
        "tpe_result_specs": {
            "FD001_fixed": {"result_filename": "same.json"}
        },
    }
    try:
        planned_result_paths(tmp_path, plan)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Duplicate planned result was accepted")


def test_receipt_payload_requires_exact_planned_result_set(tmp_path):
    plan = {
        "cap_result_specs": {
            "FD001": {
                "125": {"result_filename": "planned.json"}
            }
        },
        "tpe_result_specs": {},
    }
    try:
        receipt_integrity_payload(
            {"FD001": {"125": str(tmp_path / "results" / "extra.json")}},
            {},
            tmp_path,
            plan,
            tmp_path / "TEST_EVALUATION_STARTED.json",
        )
    except RuntimeError as exc:
        assert "differs from immutable plan" in str(exc)
    else:
        raise AssertionError("Unexpected final result set was accepted")


def test_completed_chain_rejects_result_tampering(tmp_path):
    output = tmp_path / "campaign"
    result_path = output / "results" / "FD001_locked.json"
    result_path.parent.mkdir(parents=True)
    started = output / "TEST_EVALUATION_STARTED.json"
    started.write_text('{"started": true}', encoding="utf-8")
    spec = official_spec(result_path.name)
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "canonical_test_data": {"FD001": canonical()},
        "cap_result_specs": {"FD001": {"125": spec}},
        "tpe_result_specs": {},
    }
    data_receipt = bind_official_boundary(output, plan, started)
    ensure_launch_authorization(
        output, result_path, spec, plan, started
    )
    environment = execution_environment(
        output, result_path, spec, plan, started
    )
    record_successful_attempt(
        output, result_path, spec, plan, started
    )
    result = official_result()
    result["official_execution_attestation"] = {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": environment["CCV2_TEST_PLAN_SHA256"],
        "start_marker_sha256": environment[
            "CCV2_START_MARKER_SHA256"
        ],
        "launch_authorization_sha256": environment[
            "CCV2_LAUNCH_SHA256"
        ],
        "launch_nonce": environment["CCV2_LAUNCH_NONCE"],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    write_result_receipt(output, result_path, spec, plan, started)

    result_paths = {"FD001": {"125": str(result_path)}}
    result_hashes, data_hashes = result_integrity_payload(
        result_paths, {}
    )
    receipt_hashes = receipt_integrity_payload(
        result_paths, {}, output, plan, started
    )
    summary = {
        "protocol": "confirmatory-final-summary-v2",
        "campaign_id": "screening-test",
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": sha256_file(started),
        "result_paths": result_paths,
        "tpe_result_paths": {},
        "result_file_sha256": result_hashes,
        "receipt_file_sha256": receipt_hashes,
        "test_data_verification_receipt_sha256": sha256_file(
            data_receipt
        ),
        "test_data_fingerprints": data_hashes,
        "completed_unix": 123.0,
    }
    marker = {
        "protocol": "confirmatory-final-completion-marker-v2",
        "campaign_id": "screening-test",
        "final_summary_sha256": stable_hash(summary),
        "completed_unix": 123.0,
    }
    validate_summary_chain(summary, plan, output, started)
    validate_completed_chain(
        summary, marker, plan, output, started
    )

    result["train_cap"] = 115
    result_path.write_text(json.dumps(result), encoding="utf-8")
    try:
        validate_completed_chain(
            summary, marker, plan, output, started
        )
    except RuntimeError as exc:
        assert "changed after summary" in str(exc)
    else:
        raise AssertionError("Tampered final result passed the hash chain")


def test_plan_time_fingerprints_use_only_trusted_manifest(
    tmp_path, monkeypatch
):
    trusted = {
        "protocol": "canonical-cmapss-source-v2",
        "source_page": "source",
        "source_archive": "archive",
        "source_archive_sha256": "a" * 64,
        "nested_archive_sha256": "b" * 64,
        "files": {},
        "ordered_test_targets": {},
    }
    for dataset in final_runner.DATASETS:
        trusted["files"][f"test_{dataset}.txt"] = "c" * 64
        trusted["files"][f"RUL_{dataset}.txt"] = "d" * 64
        trusted["ordered_test_targets"][dataset] = {
            "test_unit_count": 1,
            "ordered_unit_target_sha256": "e" * 64,
        }
    manifest_path = tmp_path / "canonical.json"
    manifest_path.write_text(json.dumps(trusted), encoding="utf-8")
    monkeypatch.setattr(final_runner, "CANONICAL_CMAPSS", manifest_path)
    values, source = final_runner.trusted_test_fingerprints()
    assert set(values) == set(final_runner.DATASETS)
    assert source["canonical_manifest_sha256"] == sha256_file(manifest_path)


def test_local_test_verification_refuses_pre_marker_access(tmp_path):
    plan = {
        "campaign_id": "campaign",
        "canonical_test_data": {},
        "canonical_source": {"canonical_manifest_sha256": "c" * 64},
    }
    with pytest.raises(RuntimeError, match="before the test marker"):
        final_runner.test_data_verification_core(
            tmp_path / "data",
            tmp_path / "campaign",
            plan,
            tmp_path / "campaign" / "TEST_EVALUATION_STARTED.json",
        )


def test_final_main_has_no_undefined_screening_reference():
    tree = ast.parse(
        Path(final_runner.__file__).read_text(encoding="utf-8")
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    loaded_names = {
        node.id
        for node in ast.walk(main)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "screening" not in loaded_names
