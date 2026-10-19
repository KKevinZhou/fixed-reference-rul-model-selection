import copy
import hashlib
import json
import math
import statistics

from verify_release import (
    per_seed_payload_valid,
    stable_hash,
    verify,
    verify_carrier_attempt_audit,
)


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(value, str):
        value = json.dumps(value, indent=2)
    path.write_text(value, encoding="utf-8")
    return path


def test_verify_accepts_complete_carrier_retry_chain():
    class FakeRelease:
        def __init__(self, values):
            self.values = values
            self.names = set(values)

        def read(self, name):
            return self.values[name]

        def json(self, name):
            return json.loads(self.read(name))

    plan_name = "confirmatory_campaign/carrier_transfer_plan.json"
    carrier_summary_name = (
        "confirmatory_campaign/carrier_transfer_summary.json"
    )
    original_name = (
        "confirmatory_campaign/carrier_transfer/logs/"
        "FD002_cap110_carrier.log"
    )
    retry_name = (
        "confirmatory_campaign/carrier_transfer/logs/"
        "FD002_cap110_carrier_retry1.log"
    )
    result_name = (
        "confirmatory_campaign/carrier_transfer/results/result.json"
    )
    manifest_name = (
        "confirmatory_campaign/carrier_transfer_retry_manifest.json"
    )
    retry_summary_name = (
        "confirmatory_campaign/carrier_transfer_retry_summary.json"
    )
    original_source = r"E:\campaign\carrier_transfer\logs\FD002_cap110_carrier.log"
    retry_source = (
        r"E:\campaign\carrier_transfer\logs\FD002_cap110_carrier_retry1.log"
    )
    result_source = (
        r"E:\campaign\carrier_transfer\results\result.json"
    )
    values = {
        plan_name: json.dumps(
            {
                "campaign_id": "carrier",
                "datasets": ["FD002"],
                "caps": [110],
                "trainer_sha256": "a" * 64,
                "model_sha256": "b" * 64,
            }
        ).encode(),
        original_name: (
            b"FloatingPointError: Non-finite training loss at epoch 4\n"
        ),
        retry_name: b"retry completed\n",
        result_name: b'{"status":"complete"}\n',
        "retry_carrier_transfer_failures.py": b"helper\n",
    }
    values[carrier_summary_name] = json.dumps(
        {
            "result_sha256": {
                result_source: digest(values[result_name])
            }
        }
    ).encode()
    manifest = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": "carrier",
        "carrier_plan_sha256": digest(values[plan_name]),
        "retry_helper_sha256": digest(
            values["retry_carrier_transfer_failures.py"]
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
                "failure_signature": (
                    "FloatingPointError: Non-finite training loss"
                ),
                "original_log": original_source,
                "original_log_sha256": digest(values[original_name]),
            }
        ],
    }
    values[manifest_name] = json.dumps(manifest).encode()
    summary = {
        "protocol": "carrier-transfer-exact-retry-v1",
        "campaign_id": "carrier",
        "manifest_sha256": digest(values[manifest_name]),
        "status": "complete",
        "test_evaluated": False,
        "outcomes": [
            {
                "dataset": "FD002",
                "cap": 110,
                "returncode": 0,
                "result_valid": True,
                "retry_log": retry_source,
                "retry_log_sha256": digest(values[retry_name]),
                "result_path": result_source,
                "result_sha256": digest(values[result_name]),
            }
        ],
    }
    values[retry_summary_name] = json.dumps(summary).encode()
    release = FakeRelease(values)
    source_map = {
        original_source.lower(): {
            "archive_name": original_name,
            "sha256": digest(values[original_name]),
        },
        retry_source.lower(): {
            "archive_name": retry_name,
            "sha256": digest(values[retry_name]),
        },
        result_source.lower(): {
            "archive_name": result_name,
            "sha256": digest(values[result_name]),
        },
    }

    verify_carrier_attempt_audit(
        release,
        {
            "carrier_attempt_audit": {
                "retry_used": True,
                "terminal_failure": False,
                "original_attempt_logs": 1,
                "retry_attempt_logs": 1,
                "retry_manifest_sha256": digest(values[manifest_name]),
                "retry_summary_sha256": digest(
                    values[retry_summary_name]
                ),
            }
        },
        lambda source: source_map[source.lower()],
    )


def metric_payload(raw, original, train_cap=100.0, reporting_cap=125.0):
    candidate = [min(max(value, 0.0), train_cap) for value in raw]
    common = [min(max(value, 0.0), reporting_cap) for value in raw]
    own_targets = [
        min(max(value, 0.0), train_cap) for value in original
    ]
    reference = [
        min(max(value, 0.0), reporting_cap) for value in original
    ]

    def rmse(predictions, targets):
        return math.sqrt(
            sum(
                (prediction - target) ** 2
                for prediction, target in zip(
                    predictions, targets, strict=True
                )
            )
            / len(predictions)
        )

    def nasa_score(predictions, targets):
        total = 0.0
        for prediction, target in zip(predictions, targets, strict=True):
            error = prediction - target
            if error < 0:
                total += math.exp(min(-error / 13.0, 50.0)) - 1.0
            else:
                total += math.exp(min(error / 10.0, 50.0)) - 1.0
        return total

    def exposure(targets, cap):
        return math.sqrt(
            sum(max(target - cap, 0.0) ** 2 for target in targets)
            / len(targets)
        )

    errors = [
        prediction - target
        for prediction, target in zip(common, reference, strict=True)
    ]
    metrics = {
        "own_rmse": rmse(candidate, own_targets),
        "reference_rmse": rmse(common, reference),
        "original_rmse": rmse(common, original),
        "nasa_score_reference": nasa_score(common, reference),
        "nasa_score_original": nasa_score(common, original),
        "saturation_fraction": (
            sum(target > train_cap for target in original) / len(original)
        ),
        "exposure_bound": exposure(original, train_cap),
        "joint_support_penalty": exposure(reference, train_cap),
        "mean_error": sum(errors) / len(errors),
        "late_prediction_fraction": (
            sum(error > 0 for error in errors) / len(errors)
        ),
        "early_prediction_fraction": (
            sum(error < 0 for error in errors) / len(errors)
        ),
    }
    for threshold in (20.0, 30.0, 50.0):
        due = [target <= threshold for target in reference]
        predicted_due = [
            prediction <= threshold for prediction in common
        ]
        label = str(int(threshold))
        due_count = sum(due)
        not_due_count = len(due) - due_count
        metrics[f"delayed_action_rate_at_{label}"] = (
            sum(
                is_due and not is_predicted_due
                for is_due, is_predicted_due in zip(
                    due, predicted_due, strict=True
                )
            )
            / max(due_count, 1)
        )
        metrics[f"premature_action_rate_at_{label}"] = (
            sum(
                not is_due and is_predicted_due
                for is_due, is_predicted_due in zip(
                    due, predicted_due, strict=True
                )
            )
            / max(not_due_count, 1)
        )
    return candidate, common, metrics


def objective_payload(offset):
    units = [1, 2]
    original = [150.0, 9.0]
    rows = []
    metric_values = {}
    raw_by_seed = []
    for index, seed in enumerate(range(45, 50)):
        raw = [
            130.0 + offset + 0.25 * index,
            10.0 + offset + 0.1 * index,
        ]
        candidate, common, metrics = metric_payload(raw, original)
        rows.append(
            {
                "seed": seed,
                "units": units,
                "raw_predictions": raw,
                "candidate_support_predictions": candidate,
                "common_support_predictions": common,
                "original_targets": original,
                **metrics,
            }
        )
        raw_by_seed.append(raw)
        for name, value in metrics.items():
            metric_values.setdefault(name, []).append(value)
    summary = {
        name: {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "values": values,
        }
        for name, values in metric_values.items()
    }
    raw_ensemble = [
        statistics.median(values)
        for values in zip(*raw_by_seed, strict=True)
    ]
    ensemble_candidate, ensemble_common, ensemble_metrics = metric_payload(
        raw_ensemble, original
    )
    ensemble = {
        "units": units,
        "raw_predictions": raw_ensemble,
        "candidate_support_predictions": ensemble_candidate,
        "common_support_predictions": ensemble_common,
        "original_targets": original,
        **ensemble_metrics,
    }
    return {
        "per_seed": rows,
        "summary": summary,
        "ensemble": ensemble,
    }


def build_minimal_release(tmp_path):
    root = tmp_path / "release-v1"
    source_root = r"E:\campaign"
    paths = {
        "preflight": root / "confirmatory_campaign/preflight_report.json",
        "plan": root / "confirmatory_campaign/immutable_test_plan.json",
        "started": root / "confirmatory_campaign/TEST_EVALUATION_STARTED.json",
        "data_verified": (
            root / "confirmatory_campaign/TEST_DATA_VERIFIED.json"
        ),
        "summary": root / "confirmatory_campaign/final_test_summary.json",
        "marker": root / "confirmatory_campaign/TEST_EVALUATED",
        "result": root / "confirmatory_campaign/results/result.json",
        "launch": (
            root
            / "confirmatory_campaign/final_receipts/"
            "result.json.launch.json"
        ),
        "receipt": (
            root
            / "confirmatory_campaign/final_receipts/"
            "result.json.receipt.json"
        ),
        "attempt_start": (
            root
            / "confirmatory_campaign/final_attempts/"
            "result.json.attempt001.start.json"
        ),
        "attempt_complete": (
            root
            / "confirmatory_campaign/final_attempts/"
            "result.json.attempt001.complete.json"
        ),
        "log": (
            root
            / "confirmatory_campaign/final_logs/"
            "result.log"
        ),
        "evidence": root / "confirmatory_campaign/campaign_manifest.json",
        "selection_attempt": (
            root
            / "confirmatory_campaign/SELECTION_ATTEMPT_MANIFEST.json"
        ),
        "qa_manifest": (
            root
            / "confirmatory_campaign/qa/pytest_run_manifest.json"
        ),
        "qa_junit": (
            root / "confirmatory_campaign/qa/pytest-junit.xml"
        ),
        "premarker_inventory": (
            root
            / "confirmatory_campaign/PREMARKER_CAMPAIGN_INVENTORY.json"
        ),
    }
    static_values = {
        "run_confirmatory_final.py": "final runner\n",
        "final_attested_trainer.py": "attested trainer\n",
        "confirmatory_rul_train.py": "frozen trainer\n",
        "preflight_confirmatory.py": "preflight\n",
        "summarize_confirmatory.py": "summarizer\n",
        "generate_manuscript_assets.py": "asset generator\n",
        "canonical_cmapss_sha256.json": '{"canonical": true}\n',
        "environment-lock.txt": "locked environment\n",
        "baselines/models_v3.py": "model\n",
        "create_selection_attempt_manifest.py": "builder\n",
        "run_confirmatory_cap_sweep.py": "cap runner\n",
        "run_qa_suite.py": "qa runner\n",
    }
    for name, value in static_values.items():
        write(root, name, value)
    static_hashes = {
        name: digest((root / name).read_bytes())
        for name in static_values
    }
    source_names = {
        key: source_root + "\\" + path.relative_to(
            root / "confirmatory_campaign"
        ).as_posix().replace("/", "\\")
        for key, path in paths.items()
    }
    campaign_manifest = {
        "protocol": "confirmatory-cap-sweep-v2",
        "campaign_id": "campaign",
        "trainer_sha256": static_hashes[
            "confirmatory_rul_train.py"
        ],
        "model_sha256": static_hashes["baselines/models_v3.py"],
    }
    write(
        root,
        paths["evidence"].relative_to(root),
        campaign_manifest,
    )
    evidence_hash = digest(paths["evidence"].read_bytes())
    selection_attempt = {
        "protocol": "selection-attempt-provenance-v1",
        "source_sha256": {
            "manifest_builder": static_hashes[
                "create_selection_attempt_manifest.py"
            ],
            "canonical_cap_runner": static_hashes[
                "run_confirmatory_cap_sweep.py"
            ],
            "frozen_trainer": static_hashes[
                "confirmatory_rul_train.py"
            ],
        },
    }
    selection_attempt["manifest_sha256_without_self"] = stable_hash(
        selection_attempt
    )
    write(
        root,
        paths["selection_attempt"].relative_to(root),
        selection_attempt,
    )
    write(
        root,
        paths["qa_junit"].relative_to(root),
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
    )
    qa_manifest = {
        "protocol": "artifact-qa-receipt-v1",
        "exit_code": 0,
        "counts": {
            "tests": 1,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 1,
        },
        "junit_sha256": digest(paths["qa_junit"].read_bytes()),
        "environment_lock_sha256": static_hashes[
            "environment-lock.txt"
        ],
        "qa_runner_sha256": static_hashes["run_qa_suite.py"],
        "source_file_sha256": {
            name: value
            for name, value in static_hashes.items()
            if name.endswith(".py")
        },
    }
    write(
        root,
        paths["qa_manifest"].relative_to(root),
        qa_manifest,
    )
    premarker_inventory = {
        "protocol": "premarker-campaign-inventory-v1",
        "campaign_id": "campaign",
        "test_evidence_detected": False,
        "files": {
            "campaign_manifest.json": {
                "bytes": paths["evidence"].stat().st_size,
                "sha256": evidence_hash,
            }
        },
    }
    premarker_inventory["inventory_sha256_without_self"] = stable_hash(
        premarker_inventory
    )
    write(
        root,
        paths["premarker_inventory"].relative_to(root),
        premarker_inventory,
    )
    premarker_hashes = {
        "selection_attempt_manifest_sha256": stable_hash(
            selection_attempt
        ),
        "qa_manifest_sha256": stable_hash(qa_manifest),
        "qa_junit_sha256": digest(paths["qa_junit"].read_bytes()),
        "premarker_campaign_inventory_sha256": stable_hash(
            premarker_inventory
        ),
    }
    preflight = {
        "protocol": "confirmatory-preflight-v2",
        "status": "pass",
        "test_or_truth_files_opened": False,
        "screening_campaign_id": "campaign",
        "preflight_script_sha256": static_hashes[
            "preflight_confirmatory.py"
        ],
        "canonical_source_manifest_sha256": static_hashes[
            "canonical_cmapss_sha256.json"
        ],
        "artefact_file_sha256": {
            source_names["evidence"]: evidence_hash,
        },
        **premarker_hashes,
    }
    preflight["report_sha256_without_self"] = stable_hash(preflight)
    write(root, paths["preflight"].relative_to(root), preflight)
    result_spec = {
        "result_filename": "result.json",
        "campaign_id": "campaign",
        "dataset": "FD001",
        "tag": "locked",
        "train_cap": 100,
        "reporting_cap": 125.0,
        "split_seed": 20260724,
        "endpoint_seed": 21260727,
        "model_seeds": [45, 46, 47, 48, 49],
        "config": {"carrier": "fixture"},
        "trainer_sha256": static_hashes[
            "confirmatory_rul_train.py"
        ],
        "model_sha256": static_hashes["baselines/models_v3.py"],
        "train_sha256": "train",
    }
    canonical = {
        "test_filename": "test_FD001.txt",
        "truth_filename": "RUL_FD001.txt",
        "test_sha256": "test",
        "truth_sha256": "truth",
        "test_unit_count": 2,
        "ordered_unit_target_sha256": stable_hash(
            {
                "units": [1, 2],
                "original_targets": [150.0, 9.0],
            }
        ),
    }
    plan = {
        "protocol": "confirmatory-test-plan-v2",
        "campaign_id": "campaign",
        "cap_result_specs": {"FD001": {"100": result_spec}},
        "tpe_result_specs": {},
        "canonical_test_data": {"FD001": canonical},
        "canonical_source": {
            "canonical_manifest_sha256": static_hashes[
                "canonical_cmapss_sha256.json"
            ],
        },
        "preflight_report_sha256": stable_hash(preflight),
        "final_runner_sha256": static_hashes[
            "run_confirmatory_final.py"
        ],
        "final_attested_trainer_sha256": static_hashes[
            "final_attested_trainer.py"
        ],
        "summarizer_sha256": static_hashes[
            "summarize_confirmatory.py"
        ],
        "manuscript_asset_generator_sha256": static_hashes[
            "generate_manuscript_assets.py"
        ],
        "trainer_sha256": static_hashes[
            "confirmatory_rul_train.py"
        ],
        **premarker_hashes,
    }
    write(root, paths["plan"].relative_to(root), plan)
    write(
        root,
        paths["started"].relative_to(root),
        {
            "started": True,
            "campaign_id": "campaign",
            "test_plan_sha256": stable_hash(plan),
            "selection_changes_forbidden": True,
        },
    )
    start_hash = digest(paths["started"].read_bytes())
    write(
        root,
        paths["data_verified"].relative_to(root),
        {
            "protocol": "confirmatory-local-test-data-verification-v2",
            "campaign_id": "campaign",
            "test_plan_sha256": stable_hash(plan),
            "start_marker_sha256": start_hash,
            "canonical_manifest_sha256": static_hashes[
                "canonical_cmapss_sha256.json"
            ],
            "verified_files": {
                "FD001": {
                    "test_filename": "test_FD001.txt",
                    "test_sha256": "test",
                    "truth_filename": "RUL_FD001.txt",
                    "truth_sha256": "truth",
                }
            },
            "verified_unix": 1.0,
        },
    )
    data_verified_hash = digest(paths["data_verified"].read_bytes())
    launch = {
        "protocol": "confirmatory-final-launch-v2",
        "result_path": source_names["result"],
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": start_hash,
        "expected_result": result_spec,
        "result_absent_before_authorization": True,
        "launch_nonce": "a" * 64,
        "authorized_unix": 1.0,
    }
    write(root, paths["launch"].relative_to(root), launch)
    launch_hash = digest(paths["launch"].read_bytes())
    write(root, paths["log"].relative_to(root), "official log\n")
    command = ["python", "final_attested_trainer.py"]
    attempt_start = {
        "protocol": "confirmatory-final-attempt-start-v2",
        "attempt": 1,
        "result_path": source_names["result"],
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": start_hash,
        "launch_authorization_sha256": launch_hash,
        "command": command,
        "command_sha256": stable_hash({"command": command}),
        "log_path": source_names["log"],
        "started_unix": 1.1,
    }
    write(
        root,
        paths["attempt_start"].relative_to(root),
        attempt_start,
    )
    attempt_complete = {
        "protocol": "confirmatory-final-attempt-complete-v2",
        "attempt_start_sha256": digest(
            paths["attempt_start"].read_bytes()
        ),
        "returncode": 0,
        "log_sha256": digest(paths["log"].read_bytes()),
        "completed_unix": 1.9,
    }
    write(
        root,
        paths["attempt_complete"].relative_to(root),
        attempt_complete,
    )
    result = {
        "protocol": "confirmatory-rul-cap-v2",
        "campaign_id": "campaign",
        "dataset": "FD001",
        "tag": "locked",
        "validation_only": False,
        "test_evaluated": True,
        "train_cap": 100,
        "reporting_cap": 125.0,
        "split_seed": 20260724,
        "endpoint_seed": 21260727,
        "model_seeds": [45, 46, 47, 48, 49],
        "test_unit_count": 2,
        "config": {"carrier": "fixture"},
        "source_fingerprints": {
            "trainer_sha256": static_hashes[
                "confirmatory_rul_train.py"
            ],
            "model_sha256": static_hashes["baselines/models_v3.py"],
        },
        "data_fingerprints": {
            "train_sha256": "train",
            "test_sha256": "test",
            "truth_sha256": "truth",
        },
        "official_execution_attestation": {
            "protocol": (
                "confirmatory-final-execution-attestation-v2"
            ),
            "test_plan_sha256": stable_hash(plan),
            "start_marker_sha256": start_hash,
            "launch_authorization_sha256": launch_hash,
            "launch_nonce": "a" * 64,
        },
        "test_results": {
            "self_scaled": objective_payload(0.0),
            "fixed_reference": objective_payload(-1.0),
        }
    }
    write(root, paths["result"].relative_to(root), result)
    result_hash = digest(paths["result"].read_bytes())
    receipt = {
        "protocol": "confirmatory-final-result-receipt-v2",
        "result_path": source_names["result"],
        "result_sha256": result_hash,
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": start_hash,
        "launch_authorization_sha256": launch_hash,
        "expected_result": result_spec,
        "test_data_fingerprints": {
            "test_sha256": "test",
            "truth_sha256": "truth",
        },
        "attempt_file_sha256": {
            source_names["attempt_start"]: digest(
                paths["attempt_start"].read_bytes()
            ),
            source_names["attempt_complete"]: digest(
                paths["attempt_complete"].read_bytes()
            ),
        },
        "completed_unix": 2.0,
    }
    write(root, paths["receipt"].relative_to(root), receipt)
    receipt_hash = digest(paths["receipt"].read_bytes())
    summary = {
        "protocol": "confirmatory-final-summary-v2",
        "campaign_id": "campaign",
        "test_plan_sha256": stable_hash(plan),
        "start_marker_sha256": digest(paths["started"].read_bytes()),
        "result_file_sha256": {
            source_names["result"]: result_hash,
        },
        "receipt_file_sha256": {
            source_names["receipt"]: receipt_hash,
        },
        "test_data_verification_receipt_sha256": data_verified_hash,
    }
    write(root, paths["summary"].relative_to(root), summary)
    marker = {
        "campaign_id": "campaign",
        "final_summary_sha256": stable_hash(summary),
    }
    write(root, paths["marker"].relative_to(root), marker)
    summary_csv_names = {
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
    summary_csv_paths = {
        name: write(
            root,
            f"confirmatory_campaign/summaries/{name}",
            (
                "dataset,method\nFD001,fixed_cap_125\n"
                if name == "final_test.csv"
                else "fixture\n"
            ),
        )
        for name in summary_csv_names
    }
    write(
        root,
        "confirmatory_campaign/summaries/summary_status.json",
        {
            "protocol": "confirmatory-summary-v2",
            "complete": True,
            "bound_failures": 0,
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
            "final_test_summary_sha256": digest(
                paths["summary"].read_bytes()
            ),
            "test_evaluated_marker_sha256": digest(
                paths["marker"].read_bytes()
            ),
            "summarizer_sha256": static_hashes[
                "summarize_confirmatory.py"
            ],
            "manuscript_asset_generator_sha256": static_hashes[
                "generate_manuscript_assets.py"
            ],
            "csv_sha256": {
                name: digest(path.read_bytes())
                for name, path in summary_csv_paths.items()
            },
        },
    )

    source_map = {}
    for key, path in paths.items():
        source_map[source_names[key]] = {
            "archive_name": path.relative_to(root).as_posix(),
            "sha256": digest(path.read_bytes()),
        }
    before = {
        path.relative_to(root).as_posix(): digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    provenance = {
        "protocol": "ress-release-provenance-v1",
        "artifact_version": "1.0.0",
        "campaign_id": "campaign",
        "source_campaign_root": source_root,
        "source_to_archive": source_map,
        "preflight_report": source_map[source_names["preflight"]],
        "final_summary": source_map[source_names["summary"]],
        "test_plan": source_map[source_names["plan"]],
        "test_start_marker": source_map[source_names["started"]],
        "test_data_verification": source_map[
            source_names["data_verified"]
        ],
        "test_completion_marker": source_map[source_names["marker"]],
        "premarker_provenance": {
            "selection_attempt_manifest": source_map[
                source_names["selection_attempt"]
            ],
            "qa_manifest": source_map[source_names["qa_manifest"]],
            "qa_junit": source_map[source_names["qa_junit"]],
            "premarker_campaign_inventory": source_map[
                source_names["premarker_inventory"]
            ],
        },
        "source_bindings": {
            "preflight_report_sha256": stable_hash(preflight),
            "final_runner_sha256": static_hashes[
                "run_confirmatory_final.py"
            ],
            "final_attested_trainer_sha256": static_hashes[
                "final_attested_trainer.py"
            ],
            "summarizer_sha256": static_hashes[
                "summarize_confirmatory.py"
            ],
            "manuscript_asset_generator_sha256": static_hashes[
                "generate_manuscript_assets.py"
            ],
            "trainer_sha256": static_hashes[
                "confirmatory_rul_train.py"
            ],
            "preflight_script_sha256": static_hashes[
                "preflight_confirmatory.py"
            ],
            "canonical_source_manifest_sha256": static_hashes[
                "canonical_cmapss_sha256.json"
            ],
            "model_sha256": static_hashes["baselines/models_v3.py"],
        },
        "archive_file_sha256_before_provenance": before,
    }
    write(root, "RELEASE_PROVENANCE.json", provenance)
    manifest = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.as_posix(),
    ):
        name = path.relative_to(root).as_posix()
        manifest.append(f"{digest(path.read_bytes())}  {name}")
    write(root, "MANIFEST.sha256", "\n".join(manifest) + "\n")
    return root


def test_portable_release_verifier_accepts_remapped_chain(tmp_path):
    root = build_minimal_release(tmp_path)
    result = verify(root)
    assert result["status"] == "pass"
    assert result["campaign_id"] == "campaign"


def test_portable_release_verifier_rejects_tamper(tmp_path):
    root = build_minimal_release(tmp_path)
    target = root / "confirmatory_campaign/results/result.json"
    target.write_text('{"tampered": true}', encoding="utf-8")
    try:
        verify(root)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered release passed verification")


def test_portable_semantic_validator_rejects_recomputed_field_tampering():
    result = {
        "model_seeds": [45, 46, 47, 48, 49],
        "test_unit_count": 2,
        "train_cap": 100,
        "reporting_cap": 125.0,
        "test_results": {
            "self_scaled": objective_payload(0.0),
            "fixed_reference": objective_payload(-1.0),
        },
    }
    canonical = {
        "test_unit_count": 2,
        "ordered_unit_target_sha256": stable_hash(
            {
                "units": [1, 2],
                "original_targets": [150.0, 9.0],
            }
        ),
    }
    assert per_seed_payload_valid(result, canonical)

    mutations = []
    value = copy.deepcopy(result)
    value["test_results"]["self_scaled"]["per_seed"][0][
        "nasa_score_reference"
    ] += 1.0
    mutations.append(value)
    value = copy.deepcopy(result)
    value["test_results"]["fixed_reference"]["summary"][
        "reference_rmse"
    ]["mean"] += 1.0
    mutations.append(value)
    value = copy.deepcopy(result)
    value["test_results"]["self_scaled"]["ensemble"][
        "raw_predictions"
    ][0] -= 1.0
    mutations.append(value)
    value = copy.deepcopy(result)
    value["test_results"]["fixed_reference"]["per_seed"][3][
        "original_targets"
    ][0] -= 1.0
    mutations.append(value)

    assert all(
        not per_seed_payload_valid(value, canonical)
        for value in mutations
    )
