import hashlib
import json
from datetime import date, timedelta

import pytest

from build_release import (
    campaign_relative,
    collect_carrier_attempt_evidence,
    plan_bound_result_paths,
    stable_hash,
    validated_release_citation_metadata,
    validated_release_version,
    validate_plan_source_bindings,
    validate_premarker_provenance,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_version_requires_exact_semver_match(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (artifact / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    assert validated_release_version(artifact) == "1.0.0"

    (artifact / "VERSION").write_text("1\n", encoding="utf-8")
    try:
        validated_release_version(artifact)
    except RuntimeError as exc:
        assert "semantic" in str(exc)
    else:
        raise AssertionError("Non-semantic VERSION was accepted")

    (artifact / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (artifact / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nversion: 1.0.0-beta\n",
        encoding="utf-8",
    )
    try:
        validated_release_version(artifact)
    except RuntimeError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("CFF substring version mismatch was accepted")


def test_release_requires_finalized_zenodo_doi_and_date(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    citation = artifact / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="reserved Zenodo DOI"):
        validated_release_citation_metadata(artifact)

    today = date.today().isoformat()
    citation.write_text(
        "cff-version: 1.2.0\n"
        "version: 1.0.0\n"
        f"date-released: {today}\n"
        'doi: "10.5281/zenodo.12345678"\n',
        encoding="utf-8",
    )
    assert validated_release_citation_metadata(artifact) == {
        "date_released": today,
        "doi": "10.5281/zenodo.12345678",
    }

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    citation.write_text(
        "cff-version: 1.2.0\n"
        "version: 1.0.0\n"
        f"date-released: {tomorrow}\n"
        'doi: "10.5281/zenodo.12345678"\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="future"):
        validated_release_citation_metadata(artifact)


def test_release_accepts_only_plan_bound_results_and_rejects_tamper(tmp_path):
    planned = tmp_path / "results" / "planned.json"
    extra = tmp_path / "results" / "legacy.json"
    planned.parent.mkdir(parents=True)
    planned.write_text('{"value": 1}', encoding="utf-8")
    extra.write_text('{"value": 2}', encoding="utf-8")
    campaign = {
        "result_file_sha256": {
            str(planned.resolve()): digest(planned),
        }
    }
    selected = plan_bound_result_paths(
        campaign, {planned.resolve()}
    )
    assert selected == [planned.resolve()]
    assert extra.resolve() not in selected

    planned.write_text('{"value": 3}', encoding="utf-8")
    try:
        plan_bound_result_paths(campaign, {planned.resolve()})
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered result was accepted for release")


def test_release_rejects_extra_result_in_summary_set(tmp_path):
    planned = tmp_path / "planned.json"
    extra = tmp_path / "extra.json"
    planned.write_text("planned", encoding="utf-8")
    extra.write_text("extra", encoding="utf-8")
    campaign = {
        "result_file_sha256": {
            str(planned.resolve()): digest(planned),
            str(extra.resolve()): digest(extra),
        }
    }
    try:
        plan_bound_result_paths(campaign, {planned.resolve()})
    except RuntimeError as exc:
        assert "differs from immutable test plan" in str(exc)
    else:
        raise AssertionError("Extra result in summary set was accepted")


def test_release_rejects_campaign_evidence_outside_root(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        campaign_relative(outside, campaign)
    except RuntimeError as exc:
        assert "escapes the campaign root" in str(exc)
    else:
        raise AssertionError("Outside evidence received an archive path")


def test_release_source_bindings_reject_changed_runner(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "baselines").mkdir(parents=True)
    values = {
        "run_confirmatory_final.py": "runner\n",
        "final_attested_trainer.py": "shim\n",
        "summarize_confirmatory.py": "summarizer\n",
        "generate_manuscript_assets.py": "asset generator\n",
        "confirmatory_rul_train.py": "trainer\n",
        "preflight_confirmatory.py": "preflight\n",
        "canonical_cmapss_sha256.json": "{}\n",
        "baselines/models_v3.py": "model\n",
    }
    for name, value in values.items():
        path = artifact / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    hashes = {name: digest(artifact / name) for name in values}
    manifest = {
        "protocol": "confirmatory-cap-sweep-v2",
        "campaign_id": "campaign",
        "trainer_sha256": hashes["confirmatory_rul_train.py"],
        "model_sha256": hashes["baselines/models_v3.py"],
    }
    preflight = {
        "screening_campaign_id": "campaign",
        "preflight_script_sha256": hashes[
            "preflight_confirmatory.py"
        ],
        "canonical_source_manifest_sha256": hashes[
            "canonical_cmapss_sha256.json"
        ],
    }
    plan = {
        "campaign_id": "campaign",
        "preflight_report_sha256": stable_hash(preflight),
        "final_runner_sha256": hashes["run_confirmatory_final.py"],
        "final_attested_trainer_sha256": hashes[
            "final_attested_trainer.py"
        ],
        "summarizer_sha256": hashes["summarize_confirmatory.py"],
        "manuscript_asset_generator_sha256": hashes[
            "generate_manuscript_assets.py"
        ],
        "trainer_sha256": hashes["confirmatory_rul_train.py"],
    }
    validate_plan_source_bindings(
        artifact, plan, preflight, manifest
    )
    (artifact / "run_confirmatory_final.py").write_text(
        "changed\n", encoding="utf-8"
    )
    try:
        validate_plan_source_bindings(
            artifact, plan, preflight, manifest
        )
    except RuntimeError as exc:
        payload = str(exc)
        assert "final_runner_sha256" in payload
        assert plan["final_runner_sha256"] in payload
    else:
        raise AssertionError("Changed final runner passed source binding")


def test_release_premarker_provenance_binds_inventory_and_sources(
    tmp_path,
):
    artifact = tmp_path / "artifact"
    campaign = tmp_path / "campaign"
    (artifact / "baselines").mkdir(parents=True)
    (campaign / "qa").mkdir(parents=True)
    sources = {
        "environment-lock.txt": "locked\n",
        "run_qa_suite.py": "qa\n",
        "create_selection_attempt_manifest.py": "builder\n",
        "run_confirmatory_cap_sweep.py": "runner\n",
        "confirmatory_rul_train.py": "trainer\n",
    }
    for name, value in sources.items():
        write_path = artifact / name
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(value, encoding="utf-8")
    evidence = campaign / "campaign_manifest.json"
    evidence.write_text('{"test_evaluated": false}', encoding="utf-8")
    selection = {
        "protocol": "selection-attempt-provenance-v1",
        "source_sha256": {
            "manifest_builder": digest(
                artifact / "create_selection_attempt_manifest.py"
            ),
            "canonical_cap_runner": digest(
                artifact / "run_confirmatory_cap_sweep.py"
            ),
            "frozen_trainer": digest(
                artifact / "confirmatory_rul_train.py"
            ),
        },
    }
    selection["manifest_sha256_without_self"] = stable_hash(selection)
    selection_path = campaign / "SELECTION_ATTEMPT_MANIFEST.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    junit = campaign / "qa" / "pytest-junit.xml"
    junit.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    qa = {
        "protocol": "artifact-qa-receipt-v1",
        "exit_code": 0,
        "junit_sha256": digest(junit),
        "environment_lock_sha256": digest(
            artifact / "environment-lock.txt"
        ),
        "qa_runner_sha256": digest(artifact / "run_qa_suite.py"),
    }
    qa_path = campaign / "qa" / "pytest_run_manifest.json"
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    inventory = {
        "protocol": "premarker-campaign-inventory-v1",
        "test_evidence_detected": False,
        "files": {
            "campaign_manifest.json": {
                "bytes": evidence.stat().st_size,
                "sha256": digest(evidence),
            }
        },
    }
    inventory["inventory_sha256_without_self"] = stable_hash(inventory)
    inventory_path = campaign / "PREMARKER_CAMPAIGN_INVENTORY.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    hashes = {
        "selection_attempt_manifest_sha256": stable_hash(selection),
        "qa_manifest_sha256": stable_hash(qa),
        "qa_junit_sha256": digest(junit),
        "premarker_campaign_inventory_sha256": stable_hash(inventory),
    }
    result = validate_premarker_provenance(
        artifact, campaign, hashes, hashes
    )
    assert result["hashes"] == hashes

    evidence.write_text('{"test_evaluated": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="campaign evidence changed"):
        validate_premarker_provenance(
            artifact, campaign, hashes, hashes
        )


def test_release_packages_complete_exact_carrier_retry(tmp_path):
    artifact = tmp_path / "artifact"
    campaign = tmp_path / "campaign"
    transfer = campaign / "carrier_transfer"
    logs = transfer / "logs"
    results = transfer / "results"
    logs.mkdir(parents=True)
    results.mkdir()
    helper = artifact / "retry_carrier_transfer_failures.py"
    helper.parent.mkdir()
    helper.write_text("exact retry helper\n", encoding="utf-8")
    plan_path = campaign / "carrier_transfer_plan.json"
    plan = {
        "campaign_id": "carrier-test",
        "datasets": ["FD002"],
        "caps": [110],
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    original = logs / "FD002_cap110_carrier-test.log"
    original.write_text(
        "Traceback\nFloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )
    retry_log = logs / "FD002_cap110_carrier-test_retry1.log"
    retry_log.write_text("retry completed\n", encoding="utf-8")
    result = results / "FD002_retry.json"
    result.write_text('{"status": "complete"}\n', encoding="utf-8")
    retry_manifest = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": "carrier-test",
        "carrier_plan_sha256": digest(plan_path),
        "retry_helper_sha256": digest(helper),
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
                "original_log_sha256": digest(original),
                "failure_signature": (
                    "FloatingPointError: Non-finite training loss"
                ),
            }
        ],
    }
    retry_manifest_path = (
        campaign / "carrier_transfer_retry_manifest.json"
    )
    retry_manifest_path.write_text(
        json.dumps(retry_manifest), encoding="utf-8"
    )
    retry_summary = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": "carrier-test",
        "manifest_sha256": digest(retry_manifest_path),
        "status": "complete",
        "test_evaluated": False,
        "outcomes": [
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 0,
                "result_valid": True,
                "retry_log": str(retry_log),
                "retry_log_sha256": digest(retry_log),
                "result_path": str(result),
                "result_sha256": digest(result),
            }
        ],
    }
    (campaign / "carrier_transfer_retry_summary.json").write_text(
        json.dumps(retry_summary), encoding="utf-8"
    )
    carrier_summary = {
        "result_sha256": {str(result): digest(result)}
    }
    files = {}
    source_map = {}

    audit = collect_carrier_attempt_evidence(
        files,
        source_map,
        campaign,
        plan,
        carrier_summary,
        artifact,
    )

    assert audit["retry_used"] is True
    assert audit["original_attempt_logs"] == 1
    assert audit["retry_attempt_logs"] == 1
    assert (
        "confirmatory_campaign/carrier_transfer/logs/"
        "FD002_cap110_carrier-test_retry1.log"
    ) in files


def test_release_rejects_unrecorded_carrier_failure(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    campaign = tmp_path / "campaign"
    log = campaign / "carrier_transfer/logs/FD002_cap110_carrier.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "FloatingPointError: Non-finite training loss at epoch 4\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match="without an exact-retry audit"
    ):
        collect_carrier_attempt_evidence(
            {},
            {},
            campaign,
            {
                "campaign_id": "carrier",
                "datasets": ["FD002"],
                "caps": [110],
            },
            {"result_sha256": {}},
            artifact,
        )
