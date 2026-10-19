import json

import pytest

import preflight_confirmatory as preflight
import run_qa_suite


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_premarker_inventory_binds_all_nonvolatile_json(tmp_path):
    write_json(
        tmp_path / "campaign_manifest.json",
        {"campaign_id": "campaign", "test_evaluated": False},
    )
    write_json(
        tmp_path / "results" / "FD001_validation.json",
        {
            "validation_only": True,
            "test_evaluated": False,
            "own_rmse": 8.0,
        },
    )
    write_json(
        tmp_path / "summaries" / "volatile.json",
        {"ignored": True},
    )
    first = preflight.build_premarker_inventory(tmp_path, "campaign")
    assert first["validation_result_file_count"] == 1
    assert "summaries/volatile.json" not in first["files"]

    write_json(tmp_path / "unplanned.json", {"added": "after audit"})
    second = preflight.build_premarker_inventory(tmp_path, "campaign")
    assert second != first
    assert "unplanned.json" in second["files"]


@pytest.mark.parametrize(
    "forbidden_payload",
    [
        {"test_evaluated": True},
        {"test_results": {"rmse": 1.0}},
        {"data_fingerprints": {"test_sha256": "0" * 64}},
        {"data_fingerprints": {"truth_sha256": "0" * 64}},
    ],
)
def test_premarker_inventory_rejects_test_evidence(
    tmp_path, forbidden_payload
):
    payload = {
        "validation_only": True,
        "test_evaluated": False,
    }
    payload.update(forbidden_payload)
    write_json(tmp_path / "results" / "bad.json", payload)
    with pytest.raises(RuntimeError, match="Pre-marker"):
        preflight.build_premarker_inventory(tmp_path, "campaign")


def test_premarker_inventory_allows_plan_only_during_final_recheck(
    tmp_path,
):
    write_json(
        tmp_path / "campaign_manifest.json",
        {"campaign_id": "campaign", "test_evaluated": False},
    )
    frozen = preflight.build_premarker_inventory(tmp_path, "campaign")
    write_json(tmp_path / "immutable_test_plan.json", {"frozen": True})
    with pytest.raises(RuntimeError, match="Official-test artefact"):
        preflight.build_premarker_inventory(tmp_path, "campaign")
    assert (
        preflight.build_premarker_inventory(
            tmp_path,
            "campaign",
            allow_immutable_plan=True,
        )
        == frozen
    )


def test_selection_attempt_manifest_detects_evidence_tampering(tmp_path):
    evidence = tmp_path / "logs" / "attempt.log"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("complete\n", encoding="utf-8")
    payload = {
        "protocol": "selection-attempt-provenance-v1",
        "campaign_id": "campaign",
        "selection_complete_before_manifest": True,
        "selection_changed_by_manifest": False,
        "test_evaluated": False,
        "orchestration_helper_source_archived": False,
        "source_sha256": {
            "manifest_builder": preflight.sha256_file(
                preflight.SELECTION_ATTEMPT_BUILDER
            ),
            "canonical_cap_runner": preflight.sha256_file(
                preflight.SCRIPT_DIR / "run_confirmatory_cap_sweep.py"
            ),
            "frozen_trainer": preflight.sha256_file(preflight.TRAINER),
        },
        "evidence_files": {
            "logs/attempt.log": {
                "bytes": evidence.stat().st_size,
                "sha256": preflight.sha256_file(evidence),
            }
        },
    }
    payload["manifest_sha256_without_self"] = preflight.stable_hash(
        payload
    )
    write_json(tmp_path / "SELECTION_ATTEMPT_MANIFEST.json", payload)
    preflight.validate_selection_attempt_manifest(tmp_path, "campaign")

    evidence.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence changed"):
        preflight.validate_selection_attempt_manifest(
            tmp_path, "campaign"
        )


def test_qa_receipt_detects_junit_tampering(tmp_path):
    qa = tmp_path / "qa"
    qa.mkdir()
    junit = qa / "pytest-junit.xml"
    junit.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    payload = {
        "protocol": "artifact-qa-receipt-v1",
        "exit_code": 0,
        "counts": {
            "tests": 2,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 2,
        },
        "junit_file": junit.name,
        "junit_sha256": preflight.sha256_file(junit),
        "source_file_sha256": preflight.current_python_source_hashes(),
        "environment_lock_sha256": preflight.sha256_file(
            preflight.ENVIRONMENT_LOCK
        ),
        "qa_runner_sha256": preflight.sha256_file(preflight.QA_RUNNER),
    }
    write_json(qa / "pytest_run_manifest.json", payload)
    preflight.validate_qa_receipt(tmp_path)

    junit.write_text(
        '<testsuite tests="2" failures="1" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="QA receipt"):
        preflight.validate_qa_receipt(tmp_path)


def test_junit_counts_supports_pytest_testsuites_wrapper(tmp_path):
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        "<testsuites>"
        '<testsuite tests="7" failures="1" errors="0" skipped="2"/>'
        '<testsuite tests="3" failures="0" errors="1" skipped="0"/>'
        "</testsuites>",
        encoding="utf-8",
    )
    expected = {
        "tests": 10,
        "failures": 1,
        "errors": 1,
        "skipped": 2,
        "passed": 6,
    }
    assert preflight.junit_counts(junit) == expected
    assert run_qa_suite.junit_counts(junit) == expected
