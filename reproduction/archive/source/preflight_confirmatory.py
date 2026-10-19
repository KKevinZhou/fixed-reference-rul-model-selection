"""Fail-closed integrity audit before the official C-MAPSS test is opened.

This program reads training-data fingerprints and validation-only artefacts.
It never opens C-MAPSS test or truth files.  It reconstructs the cap and TPE
decisions from their sources, verifies the robustness runs, and writes a
hashed preflight report that the final test wrapper must accept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from xml.etree import ElementTree

import optuna
from optuna.trial import TrialState


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "confirmatory_rul_train.py"
MODEL = SCRIPT_DIR / "baselines" / "models_v3.py"
CANONICAL_CMAPSS = SCRIPT_DIR / "canonical_cmapss_sha256.json"
ENVIRONMENT_LOCK = SCRIPT_DIR / "environment-lock.txt"
SELECTION_ATTEMPT_BUILDER = (
    SCRIPT_DIR / "create_selection_attempt_manifest.py"
)
QA_RUNNER = SCRIPT_DIR / "run_qa_suite.py"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
CAPS = [90, 100, 110, 115, 125]
SPLITS = [20260724, 20260725, 20260726]
PRIMARY_SPLIT = SPLITS[0]
TPE_STUDIES = [
    ("FD001", "self_scaled"),
    ("FD001", "fixed_reference"),
    ("FD001", "fixed_cap_125"),
    ("FD002", "self_scaled"),
    ("FD002", "fixed_reference"),
    ("FD002", "fixed_cap_125"),
]
OBJECTIVE_METRICS = {
    "self_scaled": ("self_scaled", "own_rmse"),
    "fixed_reference": ("fixed_reference", "reference_rmse"),
    "fixed_cap_125": ("fixed_reference", "reference_rmse"),
}
TPE_SEARCH_SEED = 20260724
TPE_MODEL_SEED = 42
TPE_SEQ = {"FD001": 50, "FD002": 50}
TPE_HUBER = [0.5, 1.0, 1.5]
CARRIER_FAILURE_SIGNATURE = "FloatingPointError: Non-finite training loss"


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


def load_json(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing required artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def campaign_result_path(root: Path, value, label: str) -> Path:
    """Resolve an untrusted manifest path before opening it."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} path is absent")
    allowed = (root.resolve() / "results").resolve()
    path = Path(value).resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} path escapes the campaign results directory: {path}"
        ) from exc
    name = path.name.lower()
    if (
        name.startswith("test_")
        or "truth" in name
        or "rul_fd" in name
    ):
        raise RuntimeError(
            f"{label} path contains a forbidden test token: {path}"
        )
    return path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_or_validate_json(path: Path, value) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"Immutable JSON differs from evidence: {path}")
        return
    atomic_json(path, value)


def validate_selection_attempt_manifest(root: Path, campaign_id: str):
    path = root / "SELECTION_ATTEMPT_MANIFEST.json"
    value = load_json(path)
    without_self = dict(value)
    observed_self_hash = without_self.pop(
        "manifest_sha256_without_self", None
    )
    if (
        value.get("protocol") != "selection-attempt-provenance-v1"
        or value.get("campaign_id") != campaign_id
        or value.get("selection_complete_before_manifest") is not True
        or value.get("selection_changed_by_manifest") is not False
        or value.get("test_evaluated") is not False
        or value.get("orchestration_helper_source_archived") is not False
        or observed_self_hash != stable_hash(without_self)
        or value.get("source_sha256")
        != {
            "manifest_builder": sha256_file(SELECTION_ATTEMPT_BUILDER),
            "canonical_cap_runner": sha256_file(
                SCRIPT_DIR / "run_confirmatory_cap_sweep.py"
            ),
            "frozen_trainer": sha256_file(TRAINER),
        }
    ):
        raise RuntimeError("Selection-attempt provenance manifest is invalid")
    evidence = value.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        raise RuntimeError("Selection-attempt evidence file set is empty")
    for relative, expected in evidence.items():
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Selection-attempt evidence escapes campaign: {relative}"
            ) from exc
        if (
            not candidate.is_file()
            or candidate.stat().st_size != expected.get("bytes")
            or sha256_file(candidate) != expected.get("sha256")
        ):
            raise RuntimeError(
                f"Selection-attempt evidence changed: {candidate}"
            )
    return path, value


def junit_counts(path: Path):
    root = ElementTree.parse(path).getroot()
    suites = (
        [root]
        if root.tag.rsplit("}", 1)[-1] == "testsuite"
        else [
            child
            for child in root
            if child.tag.rsplit("}", 1)[-1] == "testsuite"
        ]
    )
    output = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    output["passed"] = (
        output["tests"]
        - output["failures"]
        - output["errors"]
        - output["skipped"]
    )
    return output


def current_python_source_hashes():
    return {
        path.relative_to(SCRIPT_DIR).as_posix(): sha256_file(path)
        for path in sorted(SCRIPT_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def validate_qa_receipt(root: Path):
    manifest_path = root / "qa" / "pytest_run_manifest.json"
    junit_path = root / "qa" / "pytest-junit.xml"
    value = load_json(manifest_path)
    counts = junit_counts(junit_path)
    if (
        value.get("protocol") != "artifact-qa-receipt-v1"
        or value.get("exit_code") != 0
        or value.get("counts") != counts
        or counts["failures"] != 0
        or counts["errors"] != 0
        or counts["passed"] < 1
        or value.get("junit_file") != junit_path.name
        or value.get("junit_sha256") != sha256_file(junit_path)
        or value.get("source_file_sha256")
        != current_python_source_hashes()
        or value.get("environment_lock_sha256")
        != sha256_file(ENVIRONMENT_LOCK)
        or value.get("qa_runner_sha256") != sha256_file(QA_RUNNER)
    ):
        raise RuntimeError("Artifact QA receipt is invalid or stale")
    return manifest_path, junit_path, value


PREMARKER_EXCLUDED_JSON = {
    "PREMARKER_CAMPAIGN_INVENTORY.json",
    "preflight_report.json",
    "immutable_test_plan.json",
    "TEST_EVALUATION_STARTED.json",
    "TEST_DATA_VERIFIED.json",
    "final_test_summary.json",
}
FORBIDDEN_PREMARKER_KEYS = {
    "test_results",
    "per_seed_test_rmse",
    "test_rmse_mean",
    "test_predictions",
    "test_targets",
    "test_sha256",
    "truth_sha256",
}


def walk_json_items(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key, nested
            yield from walk_json_items(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_json_items(nested)


def build_premarker_inventory(
    root: Path,
    campaign_id: str,
    *,
    allow_immutable_plan: bool = False,
):
    forbidden_names = [
        "TEST_EVALUATION_STARTED.json",
        "TEST_DATA_VERIFIED.json",
        "final_test_summary.json",
        "TEST_EVALUATED",
    ]
    if not allow_immutable_plan:
        forbidden_names.append("immutable_test_plan.json")
    forbidden_existing = [
        root / name
        for name in forbidden_names
        if (root / name).exists()
    ]
    if forbidden_existing:
        raise RuntimeError(
            "Official-test artefact exists before pre-marker inventory: "
            + ", ".join(str(path) for path in forbidden_existing)
        )
    files = {}
    result_count = 0
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if (
            path.name in PREMARKER_EXCLUDED_JSON
            or path.name.startswith("FINALIZATION_WATCHER_LOCK")
            or relative.startswith("summaries/")
        ):
            continue
        payload = load_json(path)
        is_result = (
            relative.startswith("results/")
            or relative.startswith("carrier_transfer/results/")
        )
        if is_result:
            result_count += 1
            if (
                payload.get("validation_only") is not True
                or payload.get("test_evaluated") is not False
            ):
                raise RuntimeError(
                    f"Pre-marker result is not validation-only: {path}"
                )
        for key, nested in walk_json_items(payload):
            if key in FORBIDDEN_PREMARKER_KEYS:
                raise RuntimeError(
                    f"Pre-marker JSON contains test/truth payload key "
                    f"{key!r}: {path}"
                )
            if key == "test_evaluated" and nested is True:
                raise RuntimeError(
                    f"Pre-marker JSON records test evaluation: {path}"
                )
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "validation_result": is_result,
        }
    payload = {
        "protocol": "premarker-campaign-inventory-v1",
        "campaign_id": campaign_id,
        "test_evidence_detected": False,
        "json_file_count": len(files),
        "validation_result_file_count": result_count,
        "files": files,
        "excluded_json": sorted(PREMARKER_EXCLUDED_JSON),
        "excluded_volatile_patterns": [
            "FINALIZATION_WATCHER_LOCK*.json",
            "summaries/*.json",
        ],
    }
    payload["inventory_sha256_without_self"] = stable_hash(payload)
    return payload


def create_or_validate_premarker_inventory(root: Path, campaign_id: str):
    path = root / "PREMARKER_CAMPAIGN_INVENTORY.json"
    payload = build_premarker_inventory(root, campaign_id)
    create_or_validate_json(path, payload)
    return path, payload


def close(left, right, tolerance=1e-10):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def verify_metric_summary_entry(
    entry,
    *,
    split_seed,
    model_seeds,
    selection_metric,
    expected_values,
    label,
):
    """Bind one derived robustness row to independently rebuilt predictions."""
    if not isinstance(entry, dict):
        raise RuntimeError(f"Malformed metric-summary entry: {label}")
    observed_values = entry.get("values")
    if (
        entry.get("split_seed") != split_seed
        or entry.get("model_seeds") != model_seeds
        or entry.get("selection_metric") != selection_metric
        or not isinstance(observed_values, list)
        or len(observed_values) != len(expected_values)
        or any(
            not close(observed, expected)
            for observed, expected in zip(
                observed_values, expected_values, strict=True
            )
        )
    ):
        raise RuntimeError(
            f"Metric-summary entry differs from archived predictions: {label}"
        )


def mean(values):
    return float(sum(float(value) for value in values) / len(values))


def sample_std(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(
        sum((value - centre) ** 2 for value in values)
        / (len(values) - 1)
    )


def rmse(predictions, targets):
    return math.sqrt(
        mean(
            [
                (float(prediction) - float(target)) ** 2
                for prediction, target in zip(
                    predictions, targets, strict=True
                )
            ]
        )
    )


def recompute_validation_metrics(result, path):
    """Rebuild all four target/output-ceiling cells from archived predictions."""
    train_cap = float(result["train_cap"])
    reporting_cap = float(result["reporting_cap"])
    model_seeds = result.get("model_seeds", [])
    rows = result.get("per_seed_validation", [])
    if len(rows) != len(model_seeds):
        raise RuntimeError(f"Validation seed rows are incomplete: {path}")
    endpoint_targets = {
        int(item["unit"]): float(item["original_rul"])
        for item in result.get("validation_endpoints", [])
    }
    expected_units = sorted(
        int(value) for value in result.get("validation_units", [])
    )
    values = {
        objective: {
            metric: []
            for metric in (
                "own_rmse",
                "candidate_target_common_support_rmse",
                "reference_target_candidate_support_rmse",
                "reference_rmse",
                "original_rmse",
            )
        }
        for objective in ("self_scaled", "fixed_reference")
    }
    baseline_targets = None
    for index, seed_row in enumerate(rows):
        if seed_row.get("seed") != model_seeds[index]:
            raise RuntimeError(
                f"Validation seed order differs from manifest: {path}"
            )
        for objective in ("self_scaled", "fixed_reference"):
            try:
                metrics = seed_row["checkpoints"][objective]["metrics"]
                units = metrics["units"]
                raw = [float(value) for value in metrics["raw_predictions"]]
                candidate = [
                    float(value)
                    for value in metrics[
                        "candidate_support_predictions"
                    ]
                ]
                common = [
                    float(value)
                    for value in metrics[
                        "common_support_predictions"
                    ]
                ]
                original = [
                    float(value) for value in metrics["original_targets"]
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Malformed validation prediction payload: {path}"
                ) from exc
            if (
                units != expected_units
                or len(set(units)) != len(units)
                or len({len(units), len(raw), len(candidate), len(common),
                        len(original)}) != 1
                or not units
                or not all(
                    math.isfinite(value)
                    for array in (raw, candidate, common, original)
                    for value in array
                )
            ):
                raise RuntimeError(
                    f"Invalid validation prediction arrays: {path}"
                )
            if any(
                int(unit) not in endpoint_targets
                or not close(endpoint_targets[int(unit)], target)
                for unit, target in zip(units, original, strict=True)
            ):
                raise RuntimeError(
                    f"Validation targets differ from endpoint schedule: {path}"
                )
            if baseline_targets is None:
                baseline_targets = (list(units), list(original))
            elif list(units) != baseline_targets[0] or any(
                not close(left, right)
                for left, right in zip(
                    original, baseline_targets[1], strict=True
                )
            ):
                raise RuntimeError(
                    f"Validation units/targets differ across checkpoints: {path}"
                )
            if any(
                not close(value, min(max(raw_value, 0.0), train_cap))
                for value, raw_value in zip(
                    candidate, raw, strict=True
                )
            ):
                raise RuntimeError(
                    f"Candidate-support clipping is invalid: {path}"
                )
            if any(
                not close(
                    value,
                    min(max(raw_value, 0.0), reporting_cap),
                )
                for value, raw_value in zip(common, raw, strict=True)
            ):
                raise RuntimeError(
                    f"Common-support clipping is invalid: {path}"
                )
            candidate_target = [
                min(max(value, 0.0), train_cap) for value in original
            ]
            reference_target = [
                min(max(value, 0.0), reporting_cap)
                for value in original
            ]
            rebuilt = {
                "own_rmse": rmse(candidate, candidate_target),
                "candidate_target_common_support_rmse": rmse(
                    common, candidate_target
                ),
                "reference_target_candidate_support_rmse": rmse(
                    candidate, reference_target
                ),
                "reference_rmse": rmse(common, reference_target),
                "original_rmse": rmse(common, original),
            }
            own_mse = rebuilt["own_rmse"] ** 2
            candidate_original_mse = (
                sum(
                    (prediction - target) ** 2
                    for prediction, target in zip(
                        candidate, original, strict=True
                    )
                )
                / len(original)
            )
            original_exposure_mse = (
                sum(max(target - train_cap, 0.0) ** 2 for target in original)
                / len(original)
            )
            reference_gap_mse = (
                rebuilt["reference_target_candidate_support_rmse"] ** 2
                - own_mse
            )
            reference_exposure_mse = (
                sum(
                    max(target - train_cap, 0.0) ** 2
                    for target in reference_target
                )
                / len(reference_target)
            )
            tolerance = 1e-8 * (
                1.0
                + abs(candidate_original_mse)
                + abs(own_mse)
                + abs(original_exposure_mse)
                + abs(reference_gap_mse)
                + abs(reference_exposure_mse)
            )
            if (
                candidate_original_mse - own_mse
                < original_exposure_mse - tolerance
                or reference_gap_mse
                < reference_exposure_mse - tolerance
            ):
                raise RuntimeError(
                    f"Validation MSE-gap bound failed: {path}"
                )
            for metric in ("own_rmse", "reference_rmse", "original_rmse"):
                if not close(metrics.get(metric), rebuilt[metric]):
                    raise RuntimeError(
                        f"Stored {metric} differs from predictions: {path}"
                    )
            for metric, value in rebuilt.items():
                values[objective][metric].append(value)
    rebuilt_summary = {
        objective: {
            metric: {
                "values": metric_values,
                "mean": mean(metric_values),
                "std": sample_std(metric_values),
            }
            for metric, metric_values in objective_values.items()
        }
        for objective, objective_values in values.items()
    }
    stored = result.get("validation_summary", {})
    for objective in ("self_scaled", "fixed_reference"):
        for metric in ("own_rmse", "reference_rmse", "original_rmse"):
            expected = rebuilt_summary[objective][metric]
            observed = stored.get(objective, {}).get(metric, {})
            if (
                len(observed.get("values", []))
                != len(expected["values"])
                or any(
                    not close(left, right)
                    for left, right in zip(
                        observed.get("values", []),
                        expected["values"],
                        strict=True,
                    )
                )
                or not close(observed.get("mean"), expected["mean"])
                or not close(observed.get("std"), expected["std"])
            ):
                raise RuntimeError(
                    f"Validation summary {objective}/{metric} differs "
                    f"from predictions: {path}"
                )
    return rebuilt_summary


def verify_candidate_table(observed, rebuilt, label):
    if len(observed) != len(rebuilt):
        raise RuntimeError(f"Candidate-table length differs for {label}")
    by_cap = {int(item["cap"]): item for item in observed}
    for expected in rebuilt:
        cap = int(expected["cap"])
        item = by_cap.get(cap)
        if item is None or not close(item["mean"], expected["mean"]):
            raise RuntimeError(f"Candidate-table mean differs for {label}")
        if "split_values" in expected:
            values = item.get("split_values")
            if (
                values is None
                or len(values) != len(expected["split_values"])
                or any(
                    not close(left, right)
                    for left, right in zip(
                        values, expected["split_values"]
                    )
                )
            ):
                raise RuntimeError(
                    f"Candidate split values differ for {label}"
                )


def verify_split_choices(observed, rebuilt, label):
    """Compare locked split choices without exact floating-point equality."""
    if (
        not isinstance(observed, list)
        or len(observed) != len(rebuilt)
    ):
        raise RuntimeError(f"Split-choice length differs for {label}")
    for locked, expected in zip(observed, rebuilt, strict=True):
        if (
            locked.get("split_seed") != expected["split_seed"]
            or locked.get("cap") != expected["cap"]
            or not close(
                locked.get("objective_value"),
                expected["objective_value"],
            )
        ):
            raise RuntimeError(f"Split choice differs for {label}")


def validate_validation_result(
    path,
    result,
    manifest,
    campaign_id,
    dataset,
    cap,
    split_seed,
    model_seeds,
):
    config = result.get("config", {})
    checks = {
        "protocol": result.get("protocol") == "confirmatory-rul-cap-v2",
        "campaign": result.get("campaign_id") == campaign_id,
        "dataset": result.get("dataset") == dataset,
        "cap": close(result.get("train_cap"), cap),
        "reporting_cap": close(result.get("reporting_cap"), 125),
        "split_seed": result.get("split_seed") == split_seed,
        "endpoint_seed": (
            result.get("endpoint_seed") == split_seed + 1000003
        ),
        "model_seeds": result.get("model_seeds") == model_seeds,
        "validation_only": result.get("validation_only") is True,
        "test_not_evaluated": result.get("test_evaluated") is False,
        "no_test_results": "test_results" not in result,
        "no_test_hash": (
            "test_sha256" not in result.get("data_fingerprints", {})
            and "truth_sha256" not in result.get("data_fingerprints", {})
        ),
        "trainer_hash": (
            result.get("source_fingerprints", {}).get("trainer_sha256")
            == manifest["trainer_sha256"]
        ),
        "model_hash": (
            result.get("source_fingerprints", {}).get("model_sha256")
            == manifest["model_sha256"]
        ),
        "raw_train_hash": (
            result.get("data_fingerprints", {}).get("train_sha256")
            == manifest["raw_train_sha256"][dataset]
        ),
        "epochs": config.get("epochs") == manifest["epochs"],
        "patience": config.get("patience") == manifest["patience"],
        "stride": (
            config.get("train_stride") == manifest["train_stride"]
        ),
        "validation_interval": (
            config.get("validation_interval")
            == manifest["validation_interval"]
        ),
        "carrier": config.get("carrier") == manifest["carrier"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Validation result failed {failed}: {path}"
        )
    rebuilt = recompute_validation_metrics(result, path)
    for objective, metric in (
        ("self_scaled", "own_rmse"),
        ("fixed_reference", "reference_rmse"),
    ):
        values = rebuilt[objective][metric]["values"]
        if len(values) != len(model_seeds):
            raise RuntimeError(f"Invalid objective values in {path}")
    return rebuilt


def screening_audit(root, manifest):
    expected_files = {}
    decisions = {}
    for dataset in DATASETS:
        decisions[dataset] = {}
        results = {}
        for split_seed in SPLITS:
            for cap in CAPS:
                path = (
                    root
                    / "results"
                    / f"{dataset}_cap{cap}_split{split_seed}.json"
                )
                result = load_json(path)
                rebuilt = validate_validation_result(
                    path,
                    result,
                    manifest,
                    manifest["campaign_id"],
                    dataset,
                    cap,
                    split_seed,
                    [42],
                )
                results[(cap, split_seed)] = rebuilt
                expected_files[str(path)] = sha256_file(path)
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            candidates = []
            split_choices = []
            for split_seed in SPLITS:
                split_values = []
                for cap in CAPS:
                    value = float(
                        results[(cap, split_seed)][objective][metric][
                            "mean"
                        ]
                    )
                    split_values.append((value, cap))
                value, cap = min(
                    split_values, key=lambda item: (item[0], -item[1])
                )
                split_choices.append(
                    {
                        "split_seed": split_seed,
                        "cap": cap,
                        "objective_value": value,
                    }
                )
            for cap in CAPS:
                values = [
                    float(
                        results[(cap, split_seed)][objective][metric][
                            "mean"
                        ]
                    )
                    for split_seed in SPLITS
                ]
                candidates.append(
                    {
                        "cap": cap,
                        "split_values": values,
                        "mean": mean(values),
                    }
                )
            selected = min(
                candidates, key=lambda item: (item["mean"], -item["cap"])
            )
            decisions[dataset][objective] = {
                "metric": metric,
                "selected_cap": selected["cap"],
                "selection_value": selected["mean"],
                "split_choices": split_choices,
                "candidate_table": candidates,
            }
    lock_path = root / "locked_cap_decisions.json"
    lock = load_json(lock_path)
    if (
        lock.get("protocol") != "confirmatory-cap-lock-v2"
        or lock.get("campaign_id") != manifest["campaign_id"]
        or lock.get("campaign_manifest_sha256") != stable_hash(manifest)
    ):
        raise RuntimeError("Screening lock metadata is invalid")
    for dataset in DATASETS:
        for objective in ("self_scaled", "fixed_reference"):
            observed = lock["decisions"][dataset][objective]
            rebuilt = decisions[dataset][objective]
            if (
                observed["metric"] != rebuilt["metric"]
                or observed["selected_cap"] != rebuilt["selected_cap"]
                or not close(
                    observed["selection_value"],
                    rebuilt["selection_value"],
                )
            ):
                raise RuntimeError(
                    f"Screening lock cannot be reconstructed for "
                    f"{dataset} {objective}"
                )
            verify_split_choices(
                observed["split_choices"],
                rebuilt["split_choices"],
                f"screening {dataset} {objective}",
            )
            verify_candidate_table(
                observed["candidate_table"],
                rebuilt["candidate_table"],
                f"screening {dataset} {objective}",
            )
    expected_files[str(lock_path)] = sha256_file(lock_path)
    return lock, decisions, expected_files


def final_cap_audit(root, manifest, screening_lock):
    stability_manifest_path = root / "stability_manifest.json"
    final_lock_path = root / "final_cap_decisions.json"
    stability = load_json(stability_manifest_path)
    final_lock = load_json(final_lock_path)
    if (
        stability.get("screening_campaign_id") != manifest["campaign_id"]
        or stability.get("screening_manifest_sha256")
        != stable_hash(manifest)
        or stability.get("trainer_sha256") != manifest["trainer_sha256"]
        or stability.get("stability_script_sha256")
        != sha256_file(
            SCRIPT_DIR / "run_confirmatory_seed_stability.py"
        )
        or stability.get("test_evaluated") is not False
    ):
        raise RuntimeError("Stability manifest is invalid")
    if (
        final_lock.get("protocol") != "confirmatory-final-cap-lock-v2"
        or final_lock.get("screening_campaign_id")
        != manifest["campaign_id"]
        or final_lock.get("stability_campaign_id")
        != stability["campaign_id"]
        or final_lock.get("screening_lock_sha256")
        != stable_hash(screening_lock)
        or final_lock.get("stability_manifest_sha256")
        != stable_hash(stability)
        or final_lock.get("test_evaluated") is not False
    ):
        raise RuntimeError("Final cap lock metadata is invalid")

    files = {
        str(stability_manifest_path): sha256_file(stability_manifest_path),
        str(final_lock_path): sha256_file(final_lock_path),
    }
    for dataset in DATASETS:
        candidate_tables = {}
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            candidates = []
            for cap in CAPS:
                screening_by_split = {}
                rebuilt_by_split = {}
                for split_seed in SPLITS:
                    path = (
                        root
                        / "results"
                        / f"{dataset}_cap{cap}_split{split_seed}.json"
                    )
                    screening_by_split[split_seed] = load_json(path)
                    rebuilt_by_split[split_seed] = (
                        recompute_validation_metrics(
                            screening_by_split[split_seed], path
                        )
                    )
                extra_path = (
                    root
                    / "results"
                    / (
                        f"{dataset}_stability_cap{cap}_split"
                        f"{PRIMARY_SPLIT}_seeds43-44.json"
                    )
                )
                extra = load_json(extra_path)
                extra_rebuilt = validate_validation_result(
                    extra_path,
                    extra,
                    manifest,
                    stability["campaign_id"],
                    dataset,
                    cap,
                    PRIMARY_SPLIT,
                    [43, 44],
                )
                files[str(extra_path)] = sha256_file(extra_path)
                primary_values = [
                    float(
                        rebuilt_by_split[PRIMARY_SPLIT][objective][
                            metric
                        ]["values"][0]
                    ),
                    *[
                        float(value)
                        for value in extra_rebuilt[objective][metric][
                            "values"
                        ]
                    ],
                ]
                split_values = [
                    mean(primary_values),
                    *[
                        float(
                            rebuilt_by_split[split_seed][objective][
                                metric
                            ]["values"][0]
                        )
                        for split_seed in SPLITS[1:]
                    ],
                ]
                candidates.append(
                    {
                        "cap": cap,
                        "split_values": split_values,
                        "mean": mean(split_values),
                    }
                )
            selected = min(
                candidates, key=lambda item: (item["mean"], -item["cap"])
            )
            observed = final_lock["decisions"][dataset][objective]
            if (
                observed["metric"] != metric
                or observed["selected_cap"] != selected["cap"]
                or not close(
                    observed["selection_value"], selected["mean"]
                )
            ):
                raise RuntimeError(
                    f"Final cap lock cannot be reconstructed for "
                    f"{dataset} {objective}"
                )
            verify_candidate_table(
                observed["candidate_table"],
                candidates,
                f"final {dataset} {objective}",
            )
            candidate_tables[objective] = candidates
        fixed = final_lock["decisions"][dataset]["fixed_cap_125"]
        fixed_value = next(
            item["mean"]
            for item in candidate_tables["fixed_reference"]
            if item["cap"] == 125
        )
        if (
            fixed["selected_cap"] != 125
            or not close(fixed["selection_value"], fixed_value)
        ):
            raise RuntimeError(
                f"Fixed-cap control is invalid for {dataset}"
            )
    return stability, final_lock, files


def validate_tpe_result(path, result, manifest, screening, trial):
    params = trial.params
    objective, metric = OBJECTIVE_METRICS[manifest["objective"]]
    expected_cap = (
        125
        if manifest["objective"] == "fixed_cap_125"
        else params["train_cap"]
    )
    rebuilt = validate_validation_result(
        path,
        result,
        screening,
        manifest["campaign_id"],
        manifest["dataset"],
        expected_cap,
        manifest["split_seed"],
        [manifest["model_seed"]],
    )
    config = result["config"]
    if (
        int(config["seq_len"]) != manifest["fixed_seq_len"]
        or not close(config["lr"], params["lr"])
        or int(config["batch"]) != int(params["batch"])
        or not close(config["huber_delta"], params["huber_delta"])
    ):
        raise RuntimeError(f"TPE parameters do not match result {path}")
    for attribute, checkpoint, metric_name in (
        ("own_rmse", "self_scaled", "own_rmse"),
        ("reference_rmse", "fixed_reference", "reference_rmse"),
    ):
        observed = trial.user_attrs.get(attribute)
        expected = rebuilt[checkpoint][metric_name]["mean"]
        if observed is None or not close(observed, expected):
            raise RuntimeError(
                f"TPE {attribute} does not match archived predictions: {path}"
            )
    value = float(rebuilt[objective][metric]["mean"])
    if not close(value, trial.value):
        raise RuntimeError(f"TPE objective does not match result {path}")
    return value


def validate_shared_startup_candidates(locks):
    """Verify the paired TPE arms used the same six startup candidates."""
    for dataset in ("FD001", "FD002"):
        startup = {}
        for objective in ("self_scaled", "fixed_reference"):
            rows = sorted(
                locks[(dataset, objective)]["trial_table"],
                key=lambda row: int(row["number"]),
            )
            if [int(row["number"]) for row in rows[:6]] != list(range(6)):
                raise RuntimeError(
                    f"TPE startup trial numbering is invalid: "
                    f"{dataset} {objective}"
                )
            startup[objective] = [
                row["params"] for row in rows[:6]
            ]
        if startup["self_scaled"] != startup["fixed_reference"]:
            raise RuntimeError(
                f"TPE startup candidates differ between paired arms: "
                f"{dataset}"
            )


def tpe_audit(root, screening):
    files = {}
    locks = {}
    for dataset, objective_name in TPE_STUDIES:
        manifest_path = (
            root / "tpe_manifests" / f"{dataset}_{objective_name}.json"
        )
        lock_path = (
            root / "tpe_locks" / f"{dataset}_{objective_name}.json"
        )
        manifest = load_json(manifest_path)
        lock = load_json(lock_path)
        expected_search_space = {
            "train_cap": (
                [125]
                if objective_name == "fixed_cap_125"
                else CAPS
            ),
            "lr": [2e-4, 1.2e-3, "log"],
            "batch": [64, 128],
            "huber_delta": TPE_HUBER,
        }
        if (
            manifest.get("protocol") != "confirmatory-tpe-v2"
            or manifest.get("dataset") != dataset
            or manifest.get("objective") != objective_name
            or manifest.get("trials") != 12
            or manifest.get("search_seed") != TPE_SEARCH_SEED
            or manifest.get("split_seed") != PRIMARY_SPLIT
            or manifest.get("endpoint_seed")
            != PRIMARY_SPLIT + 1000003
            or manifest.get("model_seed") != TPE_MODEL_SEED
            or manifest.get("fixed_seq_len") != TPE_SEQ[dataset]
            or manifest.get("search_space") != expected_search_space
            or manifest.get("sampler")
            != {
                "name": "Optuna TPESampler",
                "n_startup_trials": 6,
            }
            or manifest.get("trainer_sha256")
            != screening["trainer_sha256"]
            or manifest.get("model_sha256")
            != screening["model_sha256"]
            or manifest.get("tpe_script_sha256")
            != sha256_file(SCRIPT_DIR / "run_confirmatory_tpe.py")
            or manifest.get("screening_campaign_id")
            != screening["campaign_id"]
            or manifest.get("test_evaluated") is not False
            or lock.get("protocol") != "confirmatory-tpe-lock-v2"
            or lock.get("campaign_id") != manifest["campaign_id"]
            or lock.get("manifest_sha256") != stable_hash(manifest)
            or lock.get("complete_trials") != 12
            or lock.get("failed_trials") != 0
            or lock.get("running_trials") != 0
            or lock.get("waiting_trials") != 0
            or lock.get("test_evaluated") is not False
        ):
            raise RuntimeError(
                f"TPE manifest/lock metadata invalid: {dataset} "
                f"{objective_name}"
            )
        database = (
            root
            / "optuna"
            / f"{dataset}_{objective_name}_{manifest['campaign_id']}.sqlite3"
        )
        if not database.exists():
            raise RuntimeError(f"Missing Optuna database: {database}")
        study = optuna.load_study(
            study_name=manifest["campaign_id"],
            storage=f"sqlite:///{database.as_posix()}",
        )
        trials = list(study.trials)
        if len(trials) != 12 or any(
            trial.state != TrialState.COMPLETE for trial in trials
        ):
            raise RuntimeError(
                f"TPE study is not exactly 12 complete trials: {database}"
            )
        rebuilt = []
        locked_trials = {
            int(item["number"]): item
            for item in lock.get("trial_table", [])
        }
        if len(locked_trials) != 12:
            raise RuntimeError(
                f"TPE lock trial table is incomplete: {lock_path}"
            )
        for trial in trials:
            path = campaign_result_path(
                root,
                trial.user_attrs.get("result_path"),
                f"TPE {dataset} {objective_name} trial {trial.number}",
            )
            if (
                not path.exists()
                or sha256_file(path)
                != trial.user_attrs["result_sha256"]
            ):
                raise RuntimeError(
                    f"TPE result hash mismatch: {path}"
                )
            result = load_json(path)
            value = validate_tpe_result(
                path, result, manifest, screening, trial
            )
            locked_trial = locked_trials.get(trial.number)
            if (
                locked_trial is None
                or locked_trial.get("state") != "COMPLETE"
                or not close(locked_trial.get("value"), value)
                or locked_trial.get("params") != trial.params
                or locked_trial.get("user_attrs")
                != trial.user_attrs
            ):
                raise RuntimeError(
                    f"TPE lock trial row differs from database: "
                    f"{dataset} {objective_name} trial {trial.number}"
                )
            rebuilt.append((value, trial.number, trial.params, path))
            files[str(path)] = sha256_file(path)
        best = min(rebuilt, key=lambda item: (item[0], item[1]))
        if (
            lock.get("best_trial") != best[1]
            or not close(lock.get("best_value"), best[0])
            or lock.get("best_params") != best[2]
            or Path(lock.get("best_result_path")) != best[3]
        ):
            raise RuntimeError(
                f"TPE best trial cannot be reconstructed: {dataset} "
                f"{objective_name}"
            )
        files[str(manifest_path)] = sha256_file(manifest_path)
        files[str(lock_path)] = sha256_file(lock_path)
        files[str(database)] = sha256_file(database)
        locks[(dataset, objective_name)] = lock
    validate_shared_startup_candidates(locks)
    return locks, files


def robustness_audit(root, screening, locks):
    manifest_path = root / "tpe_robustness_manifest.json"
    summary_path = root / "tpe_robustness_summary.json"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    expected_lock_hashes = {
        f"{dataset}_{objective}": stable_hash(locks[(dataset, objective)])
        for dataset, objective in TPE_STUDIES
    }
    if (
        manifest.get("protocol") != "confirmatory-tpe-robustness-v2"
        or manifest.get("screening_campaign_id")
        != screening["campaign_id"]
        or manifest.get("tpe_lock_hashes") != expected_lock_hashes
        or manifest.get("selection_frozen") is not True
        or manifest.get("test_evaluated") is not False
        or manifest.get("trainer_sha256")
        != screening["trainer_sha256"]
        or manifest.get("model_sha256") != screening["model_sha256"]
        or manifest.get("robustness_script_sha256")
        != sha256_file(SCRIPT_DIR / "run_tpe_robustness.py")
        or summary.get("status") != "complete"
        or summary.get("campaign_id") != manifest["campaign_id"]
        or summary.get("manifest_sha256") != stable_hash(manifest)
        or summary.get("tpe_lock_hashes") != expected_lock_hashes
        or summary.get("selection_frozen") is not True
        or summary.get("selection_changed") is not False
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("TPE robustness metadata is invalid")
    files = {
        str(manifest_path): sha256_file(manifest_path),
        str(summary_path): sha256_file(summary_path),
    }
    for dataset, objective in TPE_STUDIES:
        key = f"{dataset}_{objective}"
        paths = [
            campaign_result_path(
                root,
                value,
                f"TPE robustness {key}",
            )
            for value in summary["result_paths"][key]
        ]
        if len(paths) != 3:
            raise RuntimeError(f"Wrong robustness matrix for {key}")
        entries = summary.get("metric_summary", {}).get(key)
        if not isinstance(entries, list) or len(entries) != len(paths):
            raise RuntimeError(
                f"Wrong robustness metric-summary matrix for {key}"
            )
        expected = [
            (PRIMARY_SPLIT, [43, 44]),
            (SPLITS[1], [42]),
            (SPLITS[2], [42]),
        ]
        params = locks[(dataset, objective)]["best_params"]
        checkpoint, selection_metric = OBJECTIVE_METRICS[objective]
        for path, entry, (split_seed, seeds) in zip(
            paths, entries, expected, strict=True
        ):
            result = load_json(path)
            rebuilt = validate_validation_result(
                path,
                result,
                screening,
                manifest["campaign_id"],
                dataset,
                params["train_cap"],
                split_seed,
                seeds,
            )
            verify_metric_summary_entry(
                entry,
                split_seed=split_seed,
                model_seeds=seeds,
                selection_metric=selection_metric,
                expected_values=rebuilt[checkpoint][selection_metric][
                    "values"
                ],
                label=f"{key} split={split_seed}",
            )
            config = result["config"]
            if (
                int(config["seq_len"]) != TPE_SEQ[dataset]
                or not close(config["lr"], params["lr"])
                or int(config["batch"]) != int(params["batch"])
                or not close(
                    config["huber_delta"], params["huber_delta"]
                )
            ):
                raise RuntimeError(
                    f"Robustness parameters do not match lock: {path}"
                )
            if (
                summary.get("result_sha256", {})
                .get(key, {})
                .get(str(path))
                != sha256_file(path)
            ):
                raise RuntimeError(
                    f"Robustness result hash is missing or wrong: {path}"
                )
            files[str(path)] = sha256_file(path)
    return manifest, summary, files


def carrier_retry_audit(root, plan, carrier_summary):
    """Bind every carrier attempt and any exact same-config retry."""
    transfer_root = root / "carrier_transfer"
    logs_root = (transfer_root / "logs").resolve()
    expected_logs = {}
    files = {}
    for dataset in plan.get("datasets", []):
        for cap in plan.get("caps", []):
            path = (
                logs_root
                / (
                    f"{dataset}_cap{int(cap)}_"
                    f"{plan.get('campaign_id')}.log"
                )
            )
            if not path.is_file() or path.parent.resolve() != logs_root:
                raise RuntimeError(
                    f"Missing carrier-transfer attempt log: {path}"
                )
            expected_logs[(dataset, int(cap))] = path.resolve()
            files[str(path.resolve())] = sha256_file(path)

    failed = {
        key: path
        for key, path in expected_logs.items()
        if CARRIER_FAILURE_SIGNATURE
        in path.read_text(encoding="utf-8", errors="replace")
    }
    manifest_path = root / "carrier_transfer_retry_manifest.json"
    summary_path = root / "carrier_transfer_retry_summary.json"
    if manifest_path.exists() != summary_path.exists():
        raise RuntimeError(
            "Carrier retry manifest and summary must both exist"
        )
    if not manifest_path.exists():
        if failed:
            raise RuntimeError(
                "Carrier failure logs exist without an exact-retry audit"
            )
        return files, set()

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    if (
        manifest.get("protocol") != "carrier-transfer-exact-retry-v1"
        or manifest.get("campaign_id") != plan.get("campaign_id")
        or manifest.get("carrier_plan_sha256")
        != sha256_file(root / "carrier_transfer_plan.json")
        or manifest.get("retry_helper_sha256")
        != sha256_file(
            SCRIPT_DIR / "retry_carrier_transfer_failures.py"
        )
        or manifest.get("trainer_sha256") != plan.get("trainer_sha256")
        or manifest.get("model_sha256") != plan.get("model_sha256")
        or manifest.get("attempt_limit_per_cell") != 1
        or manifest.get("execution_mode") != "sequential"
        or manifest.get("parameter_changes") is not False
        or manifest.get("test_files_permitted") is not False
    ):
        raise RuntimeError("Carrier retry manifest is invalid")
    terminal_failure = carrier_summary.get("status") == "terminal_failure"
    expected_retry_status = "failed" if terminal_failure else "complete"
    if (
        summary.get("protocol") != manifest.get("protocol")
        or summary.get("campaign_id") != plan.get("campaign_id")
        or summary.get("manifest_sha256") != sha256_file(manifest_path)
        or summary.get("status") != expected_retry_status
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("Carrier retry summary is incomplete or invalid")

    cells = manifest.get("cells", [])
    outcomes = summary.get("outcomes", [])
    if not cells or len(cells) != len(outcomes):
        raise RuntimeError("Carrier retry cells and outcomes differ")
    covered = set()
    result_hashes = {
        str(Path(path).resolve()): digest
        for path, digest in carrier_summary.get(
            "result_sha256", {}
        ).items()
    }
    results_root = (transfer_root / "results").resolve()
    failed_retry_keys = set()
    for cell, outcome in zip(cells, outcomes, strict=True):
        try:
            key = (cell.get("dataset"), int(cell.get("cap")))
            outcome_key = (
                outcome.get("dataset"),
                int(outcome.get("cap")),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Carrier retry identity is invalid") from exc
        original = Path(cell.get("original_log", "")).resolve()
        retry_log = Path(outcome.get("retry_log", "")).resolve()
        result = Path(outcome.get("result_path", "")).resolve()
        common_valid = (
            key != outcome_key
            or failed.get(key) != original
            or cell.get("failure_signature")
            != CARRIER_FAILURE_SIGNATURE
            or cell.get("original_log_sha256") != sha256_file(original)
            or retry_log.parent != logs_root
            or not retry_log.name.endswith("_retry1.log")
            or outcome.get("retry_log_sha256") != sha256_file(retry_log)
        )
        if common_valid:
            raise RuntimeError("Carrier retry evidence is invalid")
        if outcome.get("result_valid") is True:
            if (
                outcome.get("returncode") != 0
                or result.parent != results_root
                or outcome.get("result_sha256") != sha256_file(result)
                or result_hashes.get(str(result)) != sha256_file(result)
            ):
                raise RuntimeError("Successful carrier retry is invalid")
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
                raise RuntimeError("Failed carrier retry is invalid")
            failed_retry_keys.add(key)
        covered.add(key)
        files[str(retry_log)] = sha256_file(retry_log)
    if covered != set(failed):
        raise RuntimeError(
            "Carrier retry audit does not cover every failed attempt"
        )
    files[str(manifest_path.resolve())] = sha256_file(manifest_path)
    files[str(summary_path.resolve())] = sha256_file(summary_path)
    if terminal_failure:
        missing = {
            (row.get("dataset"), int(row.get("cap")))
            for row in carrier_summary.get("missing_cells", [])
        }
        terminal_outcomes = {
            (row.get("dataset"), int(row.get("cap")))
            for row in carrier_summary.get("failed_retry_outcomes", [])
        }
        terminal_signatures = all(
            row.get("failure_signature_recurred") is True
            for row in carrier_summary.get("failed_retry_outcomes", [])
        )
        if (
            not failed_retry_keys
            or missing != failed_retry_keys
            or terminal_outcomes != failed_retry_keys
            or not terminal_signatures
            or carrier_summary.get("retry_summary_sha256")
            != sha256_file(summary_path)
        ):
            raise RuntimeError(
                "Terminal carrier summary differs from retry failures"
            )
    elif failed_retry_keys:
        raise RuntimeError("Complete carrier summary retains failed retries")
    return files, failed_retry_keys


def carrier_transfer_audit(root, screening):
    """Rebuild and bind the validation-only second-carrier check."""
    plan_path = root / "carrier_transfer_plan.json"
    summary_path = root / "carrier_transfer_summary.json"
    plan = load_json(plan_path)
    summary = load_json(summary_path)
    datasets = ("FD001", "FD002")
    caps = (90, 100, 110, 115, 125)
    robustness_path = root / "tpe_robustness_summary.json"
    expected_plan = {
        "protocol": "confirmatory-carrier-transfer-v1",
        "role": (
            "validation-only single-split, single-seed carrier sensitivity "
            "check; cannot change the locked cap-grid, TPE, or standard-test "
            "reporting decisions"
        ),
        "datasets": list(datasets),
        "caps": list(caps),
        "reporting_cap": 125,
        "split_seed": PRIMARY_SPLIT,
        "endpoint_seed": PRIMARY_SPLIT + 1000003,
        "model_seed": 42,
        "carrier": "dualstream",
        "config": {
            "FD001": {"seq_len": 50, "lr": 5e-4},
            "FD002": {"seq_len": 50, "lr": 1e-3},
        },
        "epochs": screening["epochs"],
        "patience": screening["patience"],
        "train_stride": screening["train_stride"],
        "validation_interval": screening["validation_interval"],
        "selection_rule": (
            "minimum primary-split validation objective; exact ties prefer "
            "the larger cap"
        ),
        "primary_campaign_manifest_sha256": sha256_file(
            root / "campaign_manifest.json"
        ),
        "tpe_robustness_summary_sha256": sha256_file(robustness_path),
        "runner_sha256": sha256_file(
            SCRIPT_DIR / "run_carrier_transfer.py"
        ),
        "trainer_sha256": screening["trainer_sha256"],
        "model_sha256": screening["model_sha256"],
        "raw_train_sha256": {
            dataset: screening["raw_train_sha256"][dataset]
            for dataset in datasets
        },
        "test_files_permitted": False,
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise RuntimeError(
                f"Carrier-transfer plan field differs: {key}"
            )
    unhashed_plan = dict(plan)
    observed_campaign_id = unhashed_plan.pop("campaign_id", None)
    expected_campaign_id = (
        "carrierv1-" + stable_hash(unhashed_plan)[:16]
    )
    if observed_campaign_id != expected_campaign_id:
        raise RuntimeError("Carrier-transfer campaign id is invalid")
    terminal_failure = summary.get("status") == "terminal_failure"
    if (
        summary.get("status") not in {"complete", "terminal_failure"}
        or summary.get("protocol") != plan["protocol"]
        or summary.get("campaign_id") != plan["campaign_id"]
        or summary.get("plan_sha256") != sha256_file(plan_path)
        or summary.get("selection_frozen") is not True
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("Carrier-transfer summary metadata is invalid")
    retry_files, failed_retry_keys = carrier_retry_audit(
        root, plan, summary
    )
    if terminal_failure != bool(failed_retry_keys):
        raise RuntimeError(
            "Carrier terminal status and failed-retry set differ"
        )
    expected_keys = {
        (dataset, cap) for dataset in datasets for cap in caps
    }
    expected_result_keys = expected_keys - failed_retry_keys
    allowed = (root / "carrier_transfer" / "results").resolve()
    result_paths = {}
    for path_string, expected_hash in summary.get(
        "result_sha256", {}
    ).items():
        path = Path(path_string).resolve()
        if (
            path.parent != allowed
            or re.search(r"(?:test|truth|rul_fd)", path.name, re.I)
        ):
            raise RuntimeError(
                f"Unsafe carrier-transfer result path: {path}"
            )
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"Carrier-transfer result hash differs: {path}"
            )
        result = load_json(path)
        key = (result.get("dataset"), int(result.get("train_cap")))
        if key in result_paths:
            raise RuntimeError(
                f"Duplicate carrier-transfer result cell: {key}"
            )
        result_paths[key] = (path, result)
    if set(result_paths) != expected_result_keys:
        raise RuntimeError("Carrier-transfer result matrix is incomplete")
    transfer_manifest = dict(screening)
    transfer_manifest["carrier"] = "dualstream"
    rebuilt_selections = {}
    files = {
        str(plan_path): sha256_file(plan_path),
        str(summary_path): sha256_file(summary_path),
        **retry_files,
    }
    rebuilt_profiles = {}
    for dataset in datasets:
        profile = []
        for cap in caps:
            if (dataset, cap) not in result_paths:
                continue
            path, result = result_paths[(dataset, cap)]
            rebuilt = validate_validation_result(
                path,
                result,
                transfer_manifest,
                plan["campaign_id"],
                dataset,
                cap,
                PRIMARY_SPLIT,
                [42],
            )
            profile.append(
                {
                    "cap": cap,
                    "self_scaled_own_rmse": rebuilt["self_scaled"][
                        "own_rmse"
                    ]["mean"],
                    "self_checkpoint_reference_rmse": rebuilt[
                        "self_scaled"
                    ]["reference_rmse"]["mean"],
                    "fixed_reference_rmse": rebuilt["fixed_reference"][
                        "reference_rmse"
                    ]["mean"],
                }
            )
            files[str(path)] = sha256_file(path)
        rebuilt_profiles[dataset] = profile
        if len(profile) == len(caps):
            self_cap = min(
                profile,
                key=lambda row: (
                    row["self_scaled_own_rmse"],
                    -row["cap"],
                ),
            )["cap"]
            fixed_cap = min(
                profile,
                key=lambda row: (
                    row["fixed_reference_rmse"],
                    -row["cap"],
                ),
            )["cap"]
            rebuilt_selections[dataset] = {
                "self_scaled_selected_cap": self_cap,
                "fixed_reference_selected_cap": fixed_cap,
                "selection_changed": self_cap != fixed_cap,
                "profile": profile,
            }
    observed = summary.get("selections")
    if not isinstance(observed, dict) or set(observed) != set(
        rebuilt_selections
    ):
        raise RuntimeError("Carrier-transfer selection set differs")
    for dataset, expected in rebuilt_selections.items():
        value = observed[dataset]
        if (
            value.get("self_scaled_selected_cap")
            != expected["self_scaled_selected_cap"]
            or value.get("fixed_reference_selected_cap")
            != expected["fixed_reference_selected_cap"]
            or value.get("selection_changed")
            is not expected["selection_changed"]
        ):
            raise RuntimeError(
                f"Carrier-transfer decision differs for {dataset}"
            )
        observed_profile = {
            int(row["cap"]): row for row in value.get("profile", [])
        }
        expected_profile = {
            int(row["cap"]): row for row in expected["profile"]
        }
        if set(observed_profile) != set(expected_profile):
            raise RuntimeError(
                f"Carrier-transfer profile caps differ for {dataset}"
            )
        for cap, expected_row in expected_profile.items():
            observed_row = observed_profile[cap]
            for metric in (
                "self_scaled_own_rmse",
                "self_checkpoint_reference_rmse",
                "fixed_reference_rmse",
            ):
                if not close(observed_row.get(metric), expected_row[metric]):
                    raise RuntimeError(
                        "Carrier-transfer metric differs for "
                        f"{dataset} cap={cap} metric={metric}"
                    )
    if terminal_failure:
        observed_partial = summary.get("partial_profiles")
        if (
            not isinstance(observed_partial, dict)
            or set(observed_partial) != set(rebuilt_profiles)
        ):
            raise RuntimeError(
                "Carrier-transfer partial-profile set differs"
            )
        for dataset, expected_profile in rebuilt_profiles.items():
            observed_profile = {
                int(row["cap"]): row
                for row in observed_partial.get(dataset, [])
            }
            expected_by_cap = {
                int(row["cap"]): row for row in expected_profile
            }
            if set(observed_profile) != set(expected_by_cap):
                raise RuntimeError(
                    f"Carrier partial-profile caps differ for {dataset}"
                )
            for cap, expected_row in expected_by_cap.items():
                for metric in (
                    "self_scaled_own_rmse",
                    "self_checkpoint_reference_rmse",
                    "fixed_reference_rmse",
                ):
                    if not close(
                        observed_profile[cap].get(metric),
                        expected_row[metric],
                    ):
                        raise RuntimeError(
                            "Carrier partial-profile metric differs for "
                            f"{dataset} cap={cap} metric={metric}"
                        )
        if summary.get("terminal_reason") != (
            "at least one plan-bound cell retained non-finite training "
            "loss after one exact same-configuration sequential retry"
        ):
            raise RuntimeError("Carrier terminal reason differs")
    changed = sum(
        int(value["selection_changed"])
        for value in rebuilt_selections.values()
    )
    if summary.get("selection_changed_count") != changed:
        raise RuntimeError("Carrier-transfer changed-count differs")
    return plan, summary, files


def run_preflight(root: Path, data_root: Path):
    root = root.resolve()
    data_root = data_root.resolve()
    manifest_path = root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("protocol") != "confirmatory-cap-sweep-v2"
        or manifest.get("datasets") != DATASETS
        or manifest.get("caps") != CAPS
        or manifest.get("split_seeds") != SPLITS
        or manifest.get("reporting_cap") != 125
        or manifest.get("model_seeds") != [42]
        or manifest.get("test_evaluated") is not False
        or manifest.get("trainer_sha256") != sha256_file(TRAINER)
        or manifest.get("model_sha256") != sha256_file(MODEL)
    ):
        raise RuntimeError("Screening campaign manifest is invalid")
    raw_base = data_root / "Experiment_FD001" / "CMAPSSData"
    canonical = load_json(CANONICAL_CMAPSS)
    if canonical.get("protocol") != "canonical-cmapss-source-v2":
        raise RuntimeError("Trusted C-MAPSS source manifest is invalid")
    ordered_targets = canonical.get("ordered_test_targets", {})
    if set(ordered_targets) != set(DATASETS):
        raise RuntimeError("Canonical ordered-target anchors are incomplete")
    for dataset, anchor in ordered_targets.items():
        digest = anchor.get("ordered_unit_target_sha256", "")
        count = anchor.get("test_unit_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise RuntimeError(
                f"Canonical ordered-target anchor is invalid for {dataset}"
            )
    for dataset in DATASETS:
        raw_train = raw_base / f"train_{dataset}.txt"
        observed_train_hash = sha256_file(raw_train)
        if (
            observed_train_hash != manifest["raw_train_sha256"][dataset]
            or observed_train_hash
            != canonical.get("files", {}).get(raw_train.name)
        ):
            raise RuntimeError(
                f"Raw training data changed or is not canonical for {dataset}"
            )

    screening_lock, _, screen_files = screening_audit(root, manifest)
    _, final_lock, stability_files = final_cap_audit(
        root, manifest, screening_lock
    )
    tpe_locks, tpe_files = tpe_audit(root, manifest)
    robustness_manifest, _, robustness_files = robustness_audit(
        root, manifest, tpe_locks
    )
    carrier_plan, carrier_summary, carrier_files = carrier_transfer_audit(
        root, manifest
    )
    selection_path, selection_manifest = (
        validate_selection_attempt_manifest(
            root, manifest["campaign_id"]
        )
    )
    qa_manifest_path, qa_junit_path, qa_manifest = (
        validate_qa_receipt(root)
    )
    inventory_path, premarker_inventory = (
        create_or_validate_premarker_inventory(
            root, manifest["campaign_id"]
        )
    )
    artefact_hashes = {
        str(manifest_path): sha256_file(manifest_path),
        **screen_files,
        **stability_files,
        **tpe_files,
        **robustness_files,
        **carrier_files,
        str(selection_path): sha256_file(selection_path),
        str(qa_manifest_path): sha256_file(qa_manifest_path),
        str(qa_junit_path): sha256_file(qa_junit_path),
        str(inventory_path): sha256_file(inventory_path),
    }
    report = {
        "protocol": "confirmatory-preflight-v2",
        "status": "pass",
        "test_or_truth_files_opened": False,
        "screening_campaign_id": manifest["campaign_id"],
        "final_cap_lock_sha256": stable_hash(final_lock),
        "tpe_lock_hashes": {
            f"{dataset}_{objective}": stable_hash(
                tpe_locks[(dataset, objective)]
            )
            for dataset, objective in TPE_STUDIES
        },
        "robustness_manifest_sha256": stable_hash(
            robustness_manifest
        ),
        "carrier_transfer_plan_sha256": stable_hash(carrier_plan),
        "carrier_transfer_summary_sha256": stable_hash(carrier_summary),
        "selection_attempt_manifest_sha256": stable_hash(
            selection_manifest
        ),
        "qa_manifest_sha256": stable_hash(qa_manifest),
        "qa_junit_sha256": sha256_file(qa_junit_path),
        "qa_counts": qa_manifest["counts"],
        "premarker_campaign_inventory_sha256": stable_hash(
            premarker_inventory
        ),
        "premarker_validation_result_file_count": (
            premarker_inventory["validation_result_file_count"]
        ),
        "premarker_json_file_count": (
            premarker_inventory["json_file_count"]
        ),
        "canonical_source_manifest_sha256": sha256_file(
            CANONICAL_CMAPSS
        ),
        "canonical_source_archive_sha256": canonical[
            "source_archive_sha256"
        ],
        "artefact_file_sha256": artefact_hashes,
        "preflight_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
    }
    report["report_sha256_without_self"] = stable_hash(report)
    atomic_json(root / "preflight_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=Path, default=Path(r"E:\NASA_Lab")
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    args = parser.parse_args()
    report = run_preflight(args.output_root, args.data_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
