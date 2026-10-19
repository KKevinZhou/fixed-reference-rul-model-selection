"""Build the minimal Zenodo software archive with a SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from verify_release import verify


ARTIFACT = Path(__file__).resolve().parent
WORKSPACE = ARTIFACT.parent
DEFAULT_CONFIRMATORY = Path(r"E:\NASA_Lab\ress_confirmatory_v2")
DEFAULT_OUTPUT = WORKSPACE / (
    "candidate-dependent-target-capping-artifact-v1.0.0.zip"
)
ROOT_PREFIX = "candidate-dependent-target-capping-artifact-v"
CARRIER_FAILURE_SIGNATURE = "FloatingPointError: Non-finite training loss"

STATIC_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "VERSION",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-lock.txt",
    "environment-lock.txt",
    "build_release.py",
    "verify_release.py",
    "confirmatory_rul_train.py",
    "final_attested_trainer.py",
    "canonical_cmapss_sha256.json",
    "run_confirmatory_cap_sweep.py",
    "resume_cap_sweep_parallel.py",
    "run_confirmatory_seed_stability.py",
    "run_confirmatory_tpe.py",
    "run_tpe_robustness.py",
    "run_carrier_transfer.py",
    "retry_carrier_transfer_failures.py",
    "create_selection_attempt_manifest.py",
    "run_qa_suite.py",
    "prepare_release_metadata.py",
    "preflight_confirmatory.py",
    "run_confirmatory_final.py",
    "summarize_confirmatory.py",
    "generate_manuscript_assets.py",
    "wait_and_run_seed_stability.py",
    "wait_and_run_seed_stability_parallel.py",
    "wait_and_run_tpe_suite.py",
    "wait_and_run_tpe_robustness.py",
    "wait_and_run_carrier_transfer.py",
    "wait_and_finalize_confirmatory.py",
    "analyze_exposure_bound.py",
    "audit_public_workflow.py",
    "public_workflow_audit.json",
    "archive/exploratory_v1/README.md",
    "baselines/models_v3.py",
    "tests/test_exposure_bound.py",
    "tests/test_confirmatory_protocol.py",
    "tests/test_final_receipts.py",
    "tests/test_summary_metrics.py",
    "tests/test_manuscript_assets.py",
    "tests/test_build_release.py",
    "tests/test_finalization_watcher.py",
    "tests/test_carrier_transfer.py",
    "tests/test_carrier_retry.py",
    "tests/test_parallel_resume.py",
    "tests/test_verify_release.py",
    "tests/test_public_workflow_audit.py",
    "tests/test_submission_compliance.py",
    "tests/test_preflight_tpe.py",
    "tests/test_preflight_metric_binding.py",
    "tests/test_premarker_provenance.py",
    "tests/test_release_metadata.py",
    "tests/test_approval_flow.py",
    "tests/test_zenodo_metadata.py",
]


def validated_release_version(artifact: Path) -> str:
    version = (artifact / "VERSION").read_text(
        encoding="utf-8"
    ).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(
            f"Artifact VERSION is not semantic x.y.z: {version!r}"
        )
    citation = (artifact / "CITATION.cff").read_text(encoding="utf-8")
    matches = re.findall(
        r"(?m)^version:\s*[\"']?([^\"'\s#]+)[\"']?\s*$",
        citation,
    )
    if len(matches) != 1 or matches[0] != version:
        raise RuntimeError(
            "CITATION.cff version differs from VERSION: "
            f"VERSION={version!r}, CFF={matches!r}"
        )
    return version


def validated_release_citation_metadata(artifact: Path) -> dict:
    """Require DOI-reserved, date-finalized metadata before ZIP creation."""
    citation = (artifact / "CITATION.cff").read_text(encoding="utf-8")
    date_matches = re.findall(
        r"(?m)^date-released:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$",
        citation,
    )
    doi_matches = re.findall(
        r"(?m)^doi:\s*[\"']?(10\.5281/zenodo\.\d+)[\"']?\s*$",
        citation,
    )
    if len(date_matches) != 1 or len(doi_matches) != 1:
        raise RuntimeError(
            "CITATION.cff must contain exactly one finalized "
            "date-released and one reserved Zenodo DOI"
        )
    try:
        released = date.fromisoformat(date_matches[0])
    except ValueError as exc:
        raise RuntimeError("CITATION.cff date-released is invalid") from exc
    if released > date.today():
        raise RuntimeError(
            "CITATION.cff date-released cannot be in the future"
        )
    return {"date_released": released.isoformat(), "doi": doi_matches[0]}


def add_file(files, source: Path, archive_name: str):
    if not source.is_file():
        raise FileNotFoundError(source)
    files[archive_name.replace("\\", "/")] = source.read_bytes()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_plan_source_bindings(
    artifact: Path,
    plan: dict,
    preflight: dict,
    campaign_manifest: dict,
) -> dict[str, str]:
    """Verify that the code being packaged is the code frozen by the plan."""
    artifact = artifact.resolve()
    observed = {
        "preflight_report_sha256": stable_hash(preflight),
        "final_runner_sha256": sha256_file(
            artifact / "run_confirmatory_final.py"
        ),
        "final_attested_trainer_sha256": sha256_file(
            artifact / "final_attested_trainer.py"
        ),
        "summarizer_sha256": sha256_file(
            artifact / "summarize_confirmatory.py"
        ),
        "manuscript_asset_generator_sha256": sha256_file(
            artifact / "generate_manuscript_assets.py"
        ),
        "trainer_sha256": sha256_file(
            artifact / "confirmatory_rul_train.py"
        ),
        "preflight_script_sha256": sha256_file(
            artifact / "preflight_confirmatory.py"
        ),
        "canonical_source_manifest_sha256": sha256_file(
            artifact / "canonical_cmapss_sha256.json"
        ),
        "model_sha256": sha256_file(
            artifact / "baselines" / "models_v3.py"
        ),
    }
    expected = {
        "preflight_report_sha256": plan.get(
            "preflight_report_sha256"
        ),
        "final_runner_sha256": plan.get("final_runner_sha256"),
        "final_attested_trainer_sha256": plan.get(
            "final_attested_trainer_sha256"
        ),
        "summarizer_sha256": plan.get("summarizer_sha256"),
        "manuscript_asset_generator_sha256": plan.get(
            "manuscript_asset_generator_sha256"
        ),
        "trainer_sha256": plan.get("trainer_sha256"),
        "preflight_script_sha256": preflight.get(
            "preflight_script_sha256"
        ),
        "canonical_source_manifest_sha256": preflight.get(
            "canonical_source_manifest_sha256"
        ),
        "model_sha256": campaign_manifest.get("model_sha256"),
    }
    mismatches = {
        name: {"expected": expected[name], "observed": value}
        for name, value in observed.items()
        if expected[name] != value
    }
    if (
        campaign_manifest.get("protocol") != "confirmatory-cap-sweep-v2"
        or campaign_manifest.get("campaign_id") != plan.get("campaign_id")
        or preflight.get("screening_campaign_id") != plan.get("campaign_id")
        or campaign_manifest.get("trainer_sha256")
        != plan.get("trainer_sha256")
    ):
        mismatches["campaign_identity"] = {
            "manifest_campaign_id": campaign_manifest.get("campaign_id"),
            "preflight_campaign_id": preflight.get(
                "screening_campaign_id"
            ),
            "plan_campaign_id": plan.get("campaign_id"),
            "manifest_trainer_sha256": campaign_manifest.get(
                "trainer_sha256"
            ),
            "plan_trainer_sha256": plan.get("trainer_sha256"),
        }
    if mismatches:
        raise RuntimeError(
            "Packaged source differs from immutable test-plan bindings: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def validate_premarker_provenance(
    artifact: Path,
    confirmatory: Path,
    plan: dict,
    preflight: dict,
) -> dict:
    paths = {
        "selection_attempt_manifest": (
            confirmatory / "SELECTION_ATTEMPT_MANIFEST.json"
        ),
        "qa_manifest": (
            confirmatory / "qa" / "pytest_run_manifest.json"
        ),
        "qa_junit": confirmatory / "qa" / "pytest-junit.xml",
        "premarker_campaign_inventory": (
            confirmatory / "PREMARKER_CAMPAIGN_INVENTORY.json"
        ),
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("Pre-marker provenance evidence is incomplete")
    selection = json.loads(
        paths["selection_attempt_manifest"].read_text(encoding="utf-8")
    )
    qa = json.loads(paths["qa_manifest"].read_text(encoding="utf-8"))
    inventory = json.loads(
        paths["premarker_campaign_inventory"].read_text(
            encoding="utf-8"
        )
    )
    selection_core = dict(selection)
    selection_self = selection_core.pop(
        "manifest_sha256_without_self", None
    )
    inventory_core = dict(inventory)
    inventory_self = inventory_core.pop(
        "inventory_sha256_without_self", None
    )
    expected = {
        "selection_attempt_manifest_sha256": stable_hash(selection),
        "qa_manifest_sha256": stable_hash(qa),
        "qa_junit_sha256": sha256_file(paths["qa_junit"]),
        "premarker_campaign_inventory_sha256": stable_hash(inventory),
    }
    if (
        selection.get("protocol") != "selection-attempt-provenance-v1"
        or selection_self != stable_hash(selection_core)
        or inventory.get("protocol")
        != "premarker-campaign-inventory-v1"
        or inventory.get("test_evidence_detected") is not False
        or inventory_self != stable_hash(inventory_core)
        or qa.get("protocol") != "artifact-qa-receipt-v1"
        or qa.get("exit_code") != 0
        or qa.get("junit_sha256") != expected["qa_junit_sha256"]
        or qa.get("environment_lock_sha256")
        != sha256_file(artifact / "environment-lock.txt")
        or qa.get("qa_runner_sha256")
        != sha256_file(artifact / "run_qa_suite.py")
        or selection.get("source_sha256", {}).get("manifest_builder")
        != sha256_file(
            artifact / "create_selection_attempt_manifest.py"
        )
        or selection.get("source_sha256", {}).get(
            "canonical_cap_runner"
        )
        != sha256_file(artifact / "run_confirmatory_cap_sweep.py")
        or selection.get("source_sha256", {}).get("frozen_trainer")
        != sha256_file(artifact / "confirmatory_rul_train.py")
        or any(plan.get(key) != value for key, value in expected.items())
        or any(
            preflight.get(key) != value
            for key, value in expected.items()
        )
    ):
        raise RuntimeError(
            "Pre-marker provenance is invalid or differs from the "
            "immutable test plan"
        )
    for relative, record in inventory.get("files", {}).items():
        source = (confirmatory / relative).resolve()
        try:
            source.relative_to(confirmatory.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Pre-marker inventory path escapes campaign: {relative}"
            ) from exc
        if (
            not source.is_file()
            or source.stat().st_size != record.get("bytes")
            or sha256_file(source) != record.get("sha256")
        ):
            raise RuntimeError(
                f"Pre-marker campaign evidence changed: {source}"
            )
    return {
        "paths": paths,
        "selection": selection,
        "qa": qa,
        "inventory": inventory,
        "hashes": expected,
    }


def campaign_relative(path: Path, campaign_root: Path) -> str:
    try:
        return path.resolve().relative_to(campaign_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Campaign evidence escapes the campaign root: {path}"
        ) from exc


def add_campaign_evidence(
    files: dict,
    source_map: dict,
    source: Path,
    campaign_root: Path,
    expected_hash: str | None = None,
) -> None:
    source = source.resolve()
    if not source.is_file():
        raise RuntimeError(f"Missing campaign evidence: {source}")
    observed = sha256_file(source)
    if expected_hash is not None and observed != expected_hash:
        raise RuntimeError(f"Campaign evidence hash mismatch: {source}")
    archive_name = (
        "confirmatory_campaign/"
        + campaign_relative(source, campaign_root)
    )
    add_file(files, source, archive_name)
    source_map[str(source)] = {
        "archive_name": archive_name,
        "sha256": observed,
    }


def collect_carrier_attempt_evidence(
    files: dict,
    source_map: dict,
    confirmatory: Path,
    carrier_plan: dict,
    carrier_summary: dict,
    artifact: Path = ARTIFACT,
) -> dict:
    """Package and validate the complete carrier-attempt audit trail."""
    transfer_root = confirmatory / "carrier_transfer"
    campaign_id = carrier_plan.get("campaign_id")
    expected_logs = []
    for dataset in carrier_plan.get("datasets", []):
        for cap in carrier_plan.get("caps", []):
            expected_logs.append(
                transfer_root
                / "logs"
                / f"{dataset}_cap{int(cap)}_{campaign_id}.log"
            )
    if not expected_logs:
        raise RuntimeError("Carrier plan defines no auditable attempt cells")
    for source in expected_logs:
        add_campaign_evidence(files, source_map, source, confirmatory)

    retry_manifest_path = (
        confirmatory / "carrier_transfer_retry_manifest.json"
    )
    retry_summary_path = (
        confirmatory / "carrier_transfer_retry_summary.json"
    )
    if retry_manifest_path.exists() != retry_summary_path.exists():
        raise RuntimeError(
            "Carrier retry manifest and summary must either both exist "
            "or both be absent"
        )
    failed_logs = [
        path
        for path in expected_logs
        if CARRIER_FAILURE_SIGNATURE
        in path.read_text(encoding="utf-8", errors="replace")
    ]
    if not retry_manifest_path.exists():
        if failed_logs:
            raise RuntimeError(
                "Carrier failure logs exist without an exact-retry audit"
            )
        return {
            "retry_used": False,
            "original_attempt_logs": len(expected_logs),
            "retry_attempt_logs": 0,
        }

    retry_manifest = json.loads(
        retry_manifest_path.read_text(encoding="utf-8")
    )
    retry_summary = json.loads(
        retry_summary_path.read_text(encoding="utf-8")
    )
    helper = artifact / "retry_carrier_transfer_failures.py"
    plan_path = confirmatory / "carrier_transfer_plan.json"
    if (
        retry_manifest.get("protocol")
        != "carrier-transfer-exact-retry-v1"
        or retry_manifest.get("campaign_id") != campaign_id
        or retry_manifest.get("carrier_plan_sha256")
        != sha256_file(plan_path)
        or retry_manifest.get("retry_helper_sha256")
        != sha256_file(helper)
        or retry_manifest.get("trainer_sha256")
        != carrier_plan.get("trainer_sha256")
        or retry_manifest.get("model_sha256")
        != carrier_plan.get("model_sha256")
        or retry_manifest.get("attempt_limit_per_cell") != 1
        or retry_manifest.get("execution_mode") != "sequential"
        or retry_manifest.get("parameter_changes") is not False
        or retry_manifest.get("test_files_permitted") is not False
    ):
        raise RuntimeError("Carrier retry manifest is invalid")
    terminal_failure = carrier_summary.get("status") == "terminal_failure"
    expected_retry_status = "failed" if terminal_failure else "complete"
    if (
        retry_summary.get("protocol") != retry_manifest.get("protocol")
        or retry_summary.get("campaign_id") != campaign_id
        or retry_summary.get("manifest_sha256")
        != sha256_file(retry_manifest_path)
        or retry_summary.get("status") != expected_retry_status
        or retry_summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("Carrier retry summary is incomplete or invalid")

    cells = retry_manifest.get("cells", [])
    outcomes = retry_summary.get("outcomes", [])
    if not cells or len(cells) != len(outcomes):
        raise RuntimeError("Carrier retry cells and outcomes differ")
    failed_resolved = {path.resolve() for path in failed_logs}
    seen_original = set()
    retry_logs = []
    carrier_hashes = {
        Path(path).resolve(): digest
        for path, digest in carrier_summary.get(
            "result_sha256", {}
        ).items()
    }
    failed_retry_keys = set()
    for cell, outcome in zip(cells, outcomes, strict=True):
        identity = (cell.get("dataset"), int(cell.get("cap")))
        if identity != (
            outcome.get("dataset"),
            int(outcome.get("cap")),
        ):
            raise RuntimeError("Carrier retry outcome order/identity changed")
        original = Path(cell.get("original_log", "")).resolve()
        retry_log = Path(outcome.get("retry_log", "")).resolve()
        result = Path(outcome.get("result_path", "")).resolve()
        common_invalid = (
            cell.get("failure_signature")
            != CARRIER_FAILURE_SIGNATURE
            or original not in failed_resolved
            or cell.get("original_log_sha256") != sha256_file(original)
            or outcome.get("retry_log_sha256") != sha256_file(retry_log)
        )
        if common_invalid:
            raise RuntimeError("Carrier retry evidence failed validation")
        if outcome.get("result_valid") is True:
            if (
                outcome.get("returncode") != 0
                or outcome.get("result_sha256") != sha256_file(result)
                or carrier_hashes.get(result) != sha256_file(result)
            ):
                raise RuntimeError(
                    "Successful carrier retry evidence is invalid"
                )
        else:
            if (
                not terminal_failure
                or outcome.get("failure_signature_recurred") is not True
                or CARRIER_FAILURE_SIGNATURE
                not in retry_log.read_text(
                    encoding="utf-8", errors="replace"
                )
                or result.exists()
                or "result_sha256" in outcome
            ):
                raise RuntimeError(
                    "Failed carrier retry evidence is invalid"
                )
            failed_retry_keys.add(identity)
        seen_original.add(original)
        retry_logs.append(retry_log)
    if seen_original != failed_resolved:
        raise RuntimeError(
            "Carrier retry manifest does not cover every failed attempt"
        )
    if terminal_failure:
        missing = {
            (row.get("dataset"), int(row.get("cap")))
            for row in carrier_summary.get("missing_cells", [])
        }
        if (
            not failed_retry_keys
            or missing != failed_retry_keys
            or carrier_summary.get("retry_summary_sha256")
            != sha256_file(retry_summary_path)
        ):
            raise RuntimeError("Carrier terminal-failure evidence differs")
    elif failed_retry_keys:
        raise RuntimeError("Complete carrier summary has failed retries")

    for source in (retry_manifest_path, retry_summary_path, *retry_logs):
        add_campaign_evidence(files, source_map, source, confirmatory)
    return {
        "retry_used": True,
        "terminal_failure": terminal_failure,
        "original_attempt_logs": len(expected_logs),
        "retry_attempt_logs": len(retry_logs),
        "retry_manifest_sha256": sha256_file(retry_manifest_path),
        "retry_summary_sha256": sha256_file(retry_summary_path),
    }


def plan_bound_result_paths(
    campaign: dict, expected_results: set[Path]
) -> list[Path]:
    expected = {path.resolve() for path in expected_results}
    stored_hashes = {
        Path(path).resolve(): value
        for path, value in campaign.get("result_file_sha256", {}).items()
    }
    if set(stored_hashes) != expected:
        raise RuntimeError(
            "Final summary result set differs from immutable test plan"
        )
    for path in expected:
        if not path.is_file():
            raise RuntimeError(f"Missing plan-bound result: {path}")
        if sha256_file(path) != stored_hashes[path]:
            raise RuntimeError(f"Plan-bound result hash mismatch: {path}")
    return sorted(expected, key=str)


def collect(artifact: Path = ARTIFACT, confirmatory: Path = DEFAULT_CONFIRMATORY):
    artifact = artifact.resolve()
    confirmatory = confirmatory.resolve()
    sys.path.insert(0, str(artifact))
    from run_confirmatory_final import (  # noqa: PLC0415
        planned_result_paths,
        validate_completed_chain,
    )
    from generate_manuscript_assets import (  # noqa: PLC0415
        validate_summary_provenance,
    )

    files = {}
    source_map = {}
    version = validated_release_version(artifact)
    validated_release_citation_metadata(artifact)
    add_file(files, artifact / "build_release.py", "build_release.py")
    for relative in STATIC_FILES:
        add_file(files, artifact / relative, relative)

    analysis_dir = artifact / "analysis"
    for name in (
        "exposure_bound_summary.json",
        "exposure_bound_table.csv",
    ):
        source = analysis_dir / name
        add_file(
            files,
            source,
            f"archive/exploratory_v1/analysis/{name}",
        )

    archived = artifact / "results"
    for dataset in ("FD001", "FD002", "FD003", "FD004"):
        for cap in (90, 100, 110, 115, 125):
            source = archived / f"{dataset}_V_A_gate_cap{cap}.json"
            add_file(
                files,
                source,
                f"archive/exploratory_v1/results/{source.name}",
            )

    campaign_summary = confirmatory / "final_test_summary.json"
    if not campaign_summary.exists():
        raise RuntimeError(
            "Confirmatory campaign is incomplete; final_test_summary.json is missing"
        )
    campaign = json.loads(campaign_summary.read_text(encoding="utf-8"))
    if campaign.get("protocol") != "confirmatory-final-summary-v2":
        raise RuntimeError(f"Unexpected final protocol: {campaign}")
    if not (confirmatory / "TEST_EVALUATED").exists():
        raise RuntimeError("Confirmatory TEST_EVALUATED marker is missing")
    marker = json.loads(
        (confirmatory / "TEST_EVALUATED").read_text(encoding="utf-8")
    )
    plan_path = confirmatory / "immutable_test_plan.json"
    started_path = confirmatory / "TEST_EVALUATION_STARTED.json"
    data_verified_path = confirmatory / "TEST_DATA_VERIFIED.json"
    if (
        not plan_path.is_file()
        or not started_path.is_file()
        or not data_verified_path.is_file()
    ):
        raise RuntimeError(
            "Immutable test plan/start/data-verification record is missing"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_completed_chain(
        campaign, marker, plan, confirmatory, started_path
    )
    expected_results = planned_result_paths(confirmatory, plan)
    release_results = plan_bound_result_paths(campaign, expected_results)
    preflight = confirmatory / "preflight_report.json"
    if not preflight.exists():
        raise RuntimeError("Confirmatory preflight report is missing")
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    preflight_core = dict(preflight_value)
    preflight_self_hash = preflight_core.pop(
        "report_sha256_without_self", None
    )
    if (
        preflight_value.get("status") != "pass"
        or preflight_value.get("test_or_truth_files_opened") is not False
        or stable_hash(preflight_core) != preflight_self_hash
    ):
        raise RuntimeError("Confirmatory preflight report is invalid")
    campaign_manifest_path = confirmatory / "campaign_manifest.json"
    if not campaign_manifest_path.is_file():
        raise RuntimeError("Confirmatory campaign manifest is missing")
    campaign_manifest = json.loads(
        campaign_manifest_path.read_text(encoding="utf-8")
    )
    source_bindings = validate_plan_source_bindings(
        artifact, plan, preflight_value, campaign_manifest
    )
    premarker_provenance = validate_premarker_provenance(
        artifact,
        confirmatory,
        plan,
        preflight_value,
    )
    manifest_source_key = str(campaign_manifest_path.resolve())
    if (
        preflight_value.get("artefact_file_sha256", {}).get(
            manifest_source_key
        )
        != sha256_file(campaign_manifest_path)
    ):
        raise RuntimeError(
            "Preflight does not bind the packaged campaign manifest"
        )

    # Package every validation-only source artifact admitted by preflight.
    # This is the formal evidence set, not a directory glob.
    for path_string, expected_hash in sorted(
        preflight_value.get("artefact_file_sha256", {}).items()
    ):
        add_campaign_evidence(
            files,
            source_map,
            Path(path_string),
            confirmatory,
            expected_hash,
        )
    for relative, record in sorted(
        premarker_provenance["inventory"].get("files", {}).items()
    ):
        add_campaign_evidence(
            files,
            source_map,
            confirmatory / relative,
            confirmatory,
            record["sha256"],
        )

    control_files = (
        preflight,
        plan_path,
        started_path,
        data_verified_path,
        campaign_summary,
        confirmatory / "TEST_EVALUATED",
    )
    for source in control_files:
        add_campaign_evidence(
            files, source_map, source, confirmatory
        )

    carrier_plan = confirmatory / "carrier_transfer_plan.json"
    carrier_summary_path = confirmatory / "carrier_transfer_summary.json"
    for source in (carrier_plan, carrier_summary_path):
        add_campaign_evidence(files, source_map, source, confirmatory)
    carrier_summary = json.loads(
        carrier_summary_path.read_text(encoding="utf-8")
    )
    if (
        carrier_summary.get("status")
        not in {"complete", "terminal_failure"}
        or carrier_summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("Carrier-transfer summary is incomplete")
    for path_string, expected_hash in sorted(
        carrier_summary.get("result_sha256", {}).items()
    ):
        add_campaign_evidence(
            files,
            source_map,
            Path(path_string),
            confirmatory,
            expected_hash,
        )
    carrier_attempt_audit = collect_carrier_attempt_evidence(
        files,
        source_map,
        confirmatory,
        carrier_plan,
        carrier_summary,
        artifact,
    )

    summary_dir = confirmatory / "summaries"
    summary_status = validate_summary_provenance(summary_dir)
    summary_files = [
        "summary_status.json",
        *sorted(summary_status["csv_sha256"]),
    ]
    for name in summary_files:
        source = summary_dir / name
        add_campaign_evidence(
            files, source_map, source, confirmatory
        )

    result_hash_by_path = {
        Path(path).resolve(): value
        for path, value in campaign["result_file_sha256"].items()
    }
    for source in release_results:
        add_campaign_evidence(
            files,
            source_map,
            source,
            confirmatory,
            result_hash_by_path[source.resolve()],
        )
    expected_receipts = campaign.get("receipt_file_sha256", {})
    if not expected_receipts:
        raise RuntimeError("Final summary contains no plan-bound receipts")
    for path_string, expected_hash in expected_receipts.items():
        receipt = Path(path_string)
        if not receipt.is_file():
            raise RuntimeError(f"Missing completion receipt: {receipt}")
        observed_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
        if observed_hash != expected_hash:
            raise RuntimeError(f"Completion receipt hash mismatch: {receipt}")
        add_campaign_evidence(
            files,
            source_map,
            receipt,
            confirmatory,
            expected_hash,
        )
        launch = receipt.with_name(
            receipt.name.replace(".receipt.json", ".launch.json")
        )
        if not launch.is_file():
            raise RuntimeError(
                f"Missing plan-bound launch authorization: {launch}"
            )
        add_campaign_evidence(
            files, source_map, launch, confirmatory
        )
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        for attempt_path, attempt_hash in receipt_payload.get(
            "attempt_file_sha256", {}
        ).items():
            attempt_file = Path(attempt_path)
            if not attempt_file.is_file():
                raise RuntimeError(
                    f"Missing official-attempt ledger file: {attempt_file}"
                )
            observed_attempt_hash = hashlib.sha256(
                attempt_file.read_bytes()
            ).hexdigest()
            if observed_attempt_hash != attempt_hash:
                raise RuntimeError(
                    f"Official-attempt ledger hash mismatch: {attempt_file}"
                )
            add_campaign_evidence(
                files,
                source_map,
                attempt_file,
                confirmatory,
                attempt_hash,
            )
            attempt_payload = json.loads(
                attempt_file.read_text(encoding="utf-8")
            )
            if (
                attempt_payload.get("protocol")
                == "confirmatory-final-attempt-start-v2"
            ):
                log_file = Path(attempt_payload.get("log_path", ""))
                if not log_file.is_file():
                    raise RuntimeError(
                        f"Missing official-attempt log: {log_file}"
                    )
                add_campaign_evidence(
                    files,
                    source_map,
                    log_file,
                    confirmatory,
                )

    provenance = {
        "protocol": "ress-release-provenance-v1",
        "artifact_version": (
            version
        ),
        "campaign_id": campaign["campaign_id"],
        "source_campaign_root": str(confirmatory),
        "source_to_archive": source_map,
        "preflight_report": source_map[str(preflight.resolve())],
        "final_summary": source_map[str(campaign_summary.resolve())],
        "test_plan": source_map[str(plan_path.resolve())],
        "test_start_marker": source_map[str(started_path.resolve())],
        "test_data_verification": source_map[
            str(data_verified_path.resolve())
        ],
        "test_completion_marker": source_map[
            str((confirmatory / "TEST_EVALUATED").resolve())
        ],
        "source_bindings": source_bindings,
        "carrier_attempt_audit": carrier_attempt_audit,
        "premarker_provenance": {
            key: source_map[str(path.resolve())]
            for key, path in premarker_provenance["paths"].items()
        },
        "archive_file_sha256_before_provenance": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in sorted(files.items())
        },
    }
    files["RELEASE_PROVENANCE.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_root", type=Path, default=ARTIFACT)
    parser.add_argument(
        "--confirmatory_root",
        type=Path,
        default=DEFAULT_CONFIRMATORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    files = collect(args.artifact_root, args.confirmatory_root)
    version = validated_release_version(args.artifact_root.resolve())
    release_root = ROOT_PREFIX + version
    manifest_lines = []
    for archive_name in sorted(files):
        digest = hashlib.sha256(files[archive_name]).hexdigest()
        manifest_lines.append(f"{digest}  {archive_name}")
    files["MANIFEST.sha256"] = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=output.name + ".tmp-",
        suffix=".zip",
        dir=output.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for archive_name in sorted(files):
                info = zipfile.ZipInfo(f"{release_root}/{archive_name}")
                info.date_time = (2026, 7, 24, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, files[archive_name])
        verify(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    archive_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "archive": str(output),
                "sha256": archive_digest,
                "files": len(files),
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
