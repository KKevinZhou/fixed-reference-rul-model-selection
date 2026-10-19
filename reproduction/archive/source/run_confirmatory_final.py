"""Evaluate only the immutable cap decisions from the confirmatory campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from preflight_confirmatory import (
    build_premarker_inventory,
    run_preflight,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "confirmatory_rul_train.py"
MODEL_SOURCE = SCRIPT_DIR / "baselines" / "models_v3.py"
FINAL_ATTESTED_TRAINER = SCRIPT_DIR / "final_attested_trainer.py"
SUMMARIZER = SCRIPT_DIR / "summarize_confirmatory.py"
MANUSCRIPT_ASSET_GENERATOR = SCRIPT_DIR / "generate_manuscript_assets.py"
CANONICAL_CMAPSS = SCRIPT_DIR / "canonical_cmapss_sha256.json"
ENVIRONMENT_LOCK = SCRIPT_DIR / "environment-lock.txt"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
TPE_STUDIES = [
    ("FD001", "self_scaled"),
    ("FD001", "fixed_reference"),
    ("FD001", "fixed_cap_125"),
    ("FD002", "self_scaled"),
    ("FD002", "fixed_reference"),
    ("FD002", "fixed_cap_125"),
]
DEFAULT_SEQ = {"FD001": 50, "FD002": 50}
DEFAULT_SEQ.update({"FD003": 80, "FD004": 50})
DEFAULT_LR = {
    "FD001": 5e-4,
    "FD002": 1e-3,
    "FD003": 1e-3,
    "FD004": 5e-4,
}
DEFAULT_BATCH = 128
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_EMA_SPAN = 3
DEFAULT_VAL_FRACTION = 0.15
REPORTING_CAP = 125.0
_FINAL_PROCESS_LOCK_HANDLE = None
OFFICIAL_MODEL_SEEDS = [45, 46, 47, 48, 49]
MAX_OFFICIAL_ATTEMPTS = 2
OFFICIAL_RETRY_POLICY = (
    "at most one retry, permitted only after an interrupted attempt without "
    "a completion ledger; a completed nonzero attempt is terminal"
)
OFFICIAL_SCALAR_METRICS = (
    "own_rmse",
    "reference_rmse",
    "original_rmse",
    "nasa_score_reference",
    "nasa_score_original",
    "saturation_fraction",
    "exposure_bound",
    "joint_support_penalty",
    "mean_error",
    "late_prediction_fraction",
    "early_prediction_fraction",
    "delayed_action_rate_at_20",
    "premature_action_rate_at_20",
    "delayed_action_rate_at_30",
    "premature_action_rate_at_30",
    "delayed_action_rate_at_50",
    "premature_action_rate_at_50",
)


def carrier_transfer_gate_passes(summary: dict) -> bool:
    """Accept a complete profile or an audited, frozen terminal failure."""
    return (
        summary.get("status") in {"complete", "terminal_failure"}
        and summary.get("selection_frozen") is True
        and summary.get("test_evaluated") is False
    )


def capture_runtime_environment() -> dict:
    """Capture the trainer-visible runtime without retaining a CUDA context."""
    script = r"""
import json
import platform
import numpy as np
import pandas as pd
import sklearn
import torch

available = torch.cuda.is_available()
print(json.dumps({
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "device": "cuda" if available else "cpu",
    "device_name": torch.cuda.get_device_name(0) if available else "CPU",
    "cudnn": str(torch.backends.cudnn.version()) if available else None,
    "deterministic_algorithms": True,
}, sort_keys=True))
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Cannot fingerprint the official runtime environment: "
            + process.stderr.strip()
        )
    try:
        observed = json.loads(process.stdout.strip())
    except Exception as exc:
        raise RuntimeError(
            "Official runtime environment fingerprint is malformed"
        ) from exc
    required = {
        "python",
        "torch",
        "torch_cuda",
        "numpy",
        "pandas",
        "scikit_learn",
        "device",
        "device_name",
        "cudnn",
        "deterministic_algorithms",
    }
    if set(observed) != required:
        raise RuntimeError(
            "Official runtime environment fingerprint has unexpected keys"
        )
    return observed
OFFICIAL_ARRAY_FIELDS = (
    "units",
    "raw_predictions",
    "candidate_support_predictions",
    "common_support_predictions",
    "original_targets",
)


def stable_hash(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def acquire_single_instance_lock(path: Path):
    """Hold a non-blocking, process-wide lock for the official wrapper."""
    global _FINAL_PROCESS_LOCK_HANDLE
    if _FINAL_PROCESS_LOCK_HANDLE is not None:
        raise RuntimeError("Official test wrapper is already active")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError) as exc:
        handle.close()
        raise RuntimeError(
            "Another official test wrapper already holds the campaign lock"
        ) from exc
    _FINAL_PROCESS_LOCK_HANDLE = handle
    return handle


def release_single_instance_lock() -> None:
    """Release a lock acquired by :func:`acquire_single_instance_lock`."""
    global _FINAL_PROCESS_LOCK_HANDLE
    handle = _FINAL_PROCESS_LOCK_HANDLE
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
        _FINAL_PROCESS_LOCK_HANDLE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument("--final_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=45)
    parser.add_argument("--primary_split_seed", type=int, default=20260724)
    return parser.parse_args()


def exact_float(left, right) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-15
    except (TypeError, ValueError):
        return False


def expected_config(manifest, dataset, overrides=None):
    value = {
        "seq_len": DEFAULT_SEQ[dataset],
        "lr": DEFAULT_LR[dataset],
        "batch": DEFAULT_BATCH,
        "epochs": manifest["epochs"],
        "patience": manifest["patience"],
        "huber_delta": DEFAULT_HUBER_DELTA,
        "ema_span": DEFAULT_EMA_SPAN,
        "train_stride": manifest["train_stride"],
        "validation_interval": manifest["validation_interval"],
        "val_fraction": DEFAULT_VAL_FRACTION,
        "carrier": manifest["carrier"],
    }
    if overrides:
        value.update(overrides)
    return value


def config_matches(observed, expected) -> bool:
    if not isinstance(observed, dict):
        return False
    integer_fields = {
        "seq_len",
        "batch",
        "epochs",
        "patience",
        "ema_span",
        "train_stride",
        "validation_interval",
    }
    float_fields = {"lr", "huber_delta", "val_fraction"}
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if key in integer_fields:
            try:
                if int(observed_value) != int(expected_value):
                    return False
            except (TypeError, ValueError):
                return False
        elif key in float_fields:
            if not exact_float(observed_value, expected_value):
                return False
        elif observed_value != expected_value:
            return False
    return True


def trusted_test_fingerprints():
    """Load plan-time test anchors without opening local test/truth files."""
    if not CANONICAL_CMAPSS.is_file():
        raise RuntimeError("Trusted canonical C-MAPSS manifest is missing")
    trusted = json.loads(CANONICAL_CMAPSS.read_text(encoding="utf-8"))
    if trusted.get("protocol") != "canonical-cmapss-source-v2":
        raise RuntimeError("Canonical C-MAPSS manifest protocol is invalid")
    trusted_files = trusted.get("files", {})
    trusted_targets = trusted.get("ordered_test_targets", {})
    values = {}
    for dataset in DATASETS:
        test_filename = f"test_{dataset}.txt"
        truth_filename = f"RUL_{dataset}.txt"
        for filename in (test_filename, truth_filename):
            value = trusted_files.get(filename)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(
                    f"Canonical file fingerprint is invalid: {filename}"
                )
        target_anchor = trusted_targets.get(dataset)
        if (
            not isinstance(target_anchor, dict)
            or isinstance(target_anchor.get("test_unit_count"), bool)
            or not isinstance(target_anchor.get("test_unit_count"), int)
            or target_anchor["test_unit_count"] < 1
            or not isinstance(
                target_anchor.get("ordered_unit_target_sha256"), str
            )
            or len(target_anchor["ordered_unit_target_sha256"]) != 64
        ):
            raise RuntimeError(
                f"Canonical ordered-target anchor is invalid for {dataset}"
            )
        values[dataset] = {
            "test_filename": test_filename,
            "truth_filename": truth_filename,
            "test_sha256": trusted_files[test_filename],
            "truth_sha256": trusted_files[truth_filename],
            "test_unit_count": target_anchor["test_unit_count"],
            "ordered_unit_target_sha256": target_anchor[
                "ordered_unit_target_sha256"
            ],
        }
    source = {
        "canonical_manifest_sha256": sha256_file(CANONICAL_CMAPSS),
        "source_page": trusted["source_page"],
        "source_archive": trusted["source_archive"],
        "source_archive_sha256": trusted["source_archive_sha256"],
        "nested_archive_sha256": trusted["nested_archive_sha256"],
    }
    return values, source


def test_data_verification_path(output_root: Path) -> Path:
    return output_root / "TEST_DATA_VERIFIED.json"


def test_data_verification_core(
    data_root: Path,
    output_root: Path,
    test_plan: dict,
    started_path: Path,
) -> dict:
    """Verify local official data only after the immutable start marker exists."""
    if not started_path.is_file():
        raise RuntimeError(
            "Local test/truth files cannot be verified before the test marker"
        )
    started = json.loads(started_path.read_text(encoding="utf-8"))
    if (
        started.get("campaign_id") != test_plan.get("campaign_id")
        or started.get("test_plan_sha256") != stable_hash(test_plan)
        or started.get("selection_changes_forbidden") is not True
    ):
        raise RuntimeError("Test marker does not bind the immutable plan")
    base = data_root / "Experiment_FD001" / "CMAPSSData"
    observed = {}
    for dataset in DATASETS:
        planned = test_plan["canonical_test_data"][dataset]
        pair = {}
        for role in ("test", "truth"):
            filename = planned[f"{role}_filename"]
            path = base / filename
            if not path.is_file():
                raise RuntimeError(
                    f"Missing canonical C-MAPSS {role} file: {path}"
                )
            digest = sha256_file(path)
            if digest != planned[f"{role}_sha256"]:
                raise RuntimeError(
                    f"Local C-MAPSS {role} file differs from the plan: {path}"
                )
            pair[f"{role}_filename"] = filename
            pair[f"{role}_sha256"] = digest
        observed[dataset] = pair
    return {
        "protocol": "confirmatory-local-test-data-verification-v2",
        "campaign_id": test_plan["campaign_id"],
        "test_plan_sha256": stable_hash(test_plan),
        "start_marker_sha256": sha256_file(started_path),
        "canonical_manifest_sha256": test_plan["canonical_source"][
            "canonical_manifest_sha256"
        ],
        "verified_files": observed,
    }


def verify_local_test_fingerprints(
    data_root: Path,
    output_root: Path,
    test_plan: dict,
    started_path: Path,
) -> Path:
    path = test_data_verification_path(output_root)
    expected = test_data_verification_core(
        data_root, output_root, test_plan, started_path
    )
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        observed_core = dict(observed)
        observed_core.pop("verified_unix", None)
        if observed_core != expected:
            raise RuntimeError(
                "Existing local test-data verification differs from the plan"
            )
        return path
    value = dict(expected)
    value["verified_unix"] = time.time()
    exclusive_json(path, value)
    return path


def valid_test_data_verification(
    output_root: Path, test_plan: dict, started_path: Path
) -> bool:
    path = test_data_verification_path(output_root)
    if not path.is_file() or not started_path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_files = {
        dataset: {
            "test_filename": canonical["test_filename"],
            "test_sha256": canonical["test_sha256"],
            "truth_filename": canonical["truth_filename"],
            "truth_sha256": canonical["truth_sha256"],
        }
        for dataset, canonical in test_plan["canonical_test_data"].items()
    }
    return (
        value.get("protocol")
        == "confirmatory-local-test-data-verification-v2"
        and value.get("campaign_id") == test_plan.get("campaign_id")
        and value.get("test_plan_sha256") == stable_hash(test_plan)
        and value.get("start_marker_sha256") == sha256_file(started_path)
        and value.get("canonical_manifest_sha256")
        == test_plan["canonical_source"]["canonical_manifest_sha256"]
        and value.get("verified_files") == expected_files
    )


def validate_planned_execution_sources(test_plan: dict) -> None:
    if (
        sha256_file(TRAINER)
        != test_plan.get("trainer_sha256")
        or sha256_file(MODEL_SOURCE)
        != test_plan.get("model_sha256")
        or sha256_file(Path(__file__).resolve())
        != test_plan.get("final_runner_sha256")
        or sha256_file(FINAL_ATTESTED_TRAINER)
        != test_plan.get("final_attested_trainer_sha256")
        or sha256_file(SUMMARIZER)
        != test_plan.get("summarizer_sha256")
        or sha256_file(MANUSCRIPT_ASSET_GENERATOR)
        != test_plan.get("manuscript_asset_generator_sha256")
        or sha256_file(ENVIRONMENT_LOCK)
        != test_plan.get("environment_lock_sha256")
    ):
        raise RuntimeError(
            "Official execution source changed after the immutable plan"
        )
    if capture_runtime_environment() != test_plan.get(
        "runtime_environment"
    ):
        raise RuntimeError(
            "Official runtime environment changed after the immutable plan"
        )


def planned_execution_source_hashes(manifest: dict) -> dict:
    """Return every source fingerprint enforced after plan creation."""
    return {
        "final_runner_sha256": sha256_file(Path(__file__).resolve()),
        "final_attested_trainer_sha256": sha256_file(
            FINAL_ATTESTED_TRAINER
        ),
        "summarizer_sha256": sha256_file(SUMMARIZER),
        "manuscript_asset_generator_sha256": sha256_file(
            MANUSCRIPT_ASSET_GENERATOR
        ),
        "trainer_sha256": manifest["trainer_sha256"],
        "model_sha256": manifest["model_sha256"],
    }


def result_data_matches(result, canonical) -> bool:
    fingerprints = result.get("data_fingerprints", {})
    return (
        fingerprints.get("test_sha256") == canonical["test_sha256"]
        and fingerprints.get("truth_sha256") == canonical["truth_sha256"]
    )


def result_matches_plan_spec(result, spec, canonical) -> bool:
    """Bind a completed result semantically to its immutable plan entry."""
    source = result.get("source_fingerprints", {})
    data = result.get("data_fingerprints", {})
    return (
        result.get("protocol") == "confirmatory-rul-cap-v2"
        and result.get("validation_only") is False
        and result.get("test_evaluated") is True
        and result.get("campaign_id") == spec.get("campaign_id")
        and result.get("dataset") == spec.get("dataset")
        and result.get("tag") == spec.get("tag")
        and int(result.get("train_cap")) == int(spec.get("train_cap"))
        and exact_float(
            result.get("reporting_cap"), spec.get("reporting_cap")
        )
        and int(result.get("split_seed")) == int(spec.get("split_seed"))
        and int(result.get("endpoint_seed")) == int(spec.get("endpoint_seed"))
        and result.get("model_seeds") == spec.get("model_seeds")
        and config_matches(result.get("config"), spec.get("config"))
        and source.get("trainer_sha256") == spec.get("trainer_sha256")
        and source.get("model_sha256") == spec.get("model_sha256")
        and data.get("train_sha256") == spec.get("train_sha256")
        and result.get("environment") == spec.get("runtime_environment")
        and result_data_matches(result, canonical)
        and per_seed_prediction_payload_valid(result, canonical)
    )


def per_seed_prediction_payload_valid(result: dict, canonical: dict) -> bool:
    """Independently reconstruct every official test scalar and aggregate."""
    expected_seeds = result.get("model_seeds", [])
    expected_count = canonical.get("test_unit_count")
    canonical_target_hash = canonical.get("ordered_unit_target_sha256")
    test_results = result.get("test_results", {})
    try:
        train_cap = float(result["train_cap"])
        reporting_cap = float(result["reporting_cap"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        expected_seeds != OFFICIAL_MODEL_SEEDS
        or expected_count is None
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
        or not isinstance(canonical_target_hash, str)
        or len(canonical_target_hash) != 64
        or result.get("test_unit_count") != expected_count
        or not math.isfinite(train_cap)
        or not math.isfinite(reporting_cap)
        or train_cap <= 0
        or reporting_cap <= 0
    ):
        return False
    baseline_units = None
    baseline_targets = None

    def close(left, right, tolerance=1e-8):
        try:
            return math.isclose(
                float(left),
                float(right),
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        except (TypeError, ValueError):
            return False

    def numeric_list(value, expected_length):
        if not isinstance(value, list) or len(value) != expected_length:
            return None
        try:
            observed = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in observed):
            return None
        return observed

    def recompute_rmse(predictions, targets):
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

    def decision_metrics(predictions, targets):
        errors = [
            prediction - target
            for prediction, target in zip(predictions, targets, strict=True)
        ]
        values = {
            "mean_error": sum(errors) / len(errors),
            "late_prediction_fraction": sum(
                error > 0 for error in errors
            )
            / len(errors),
            "early_prediction_fraction": sum(
                error < 0 for error in errors
            )
            / len(errors),
        }
        for threshold in (20.0, 30.0, 50.0):
            due = [target <= threshold for target in targets]
            predicted_due = [
                prediction <= threshold for prediction in predictions
            ]
            label = str(int(threshold))
            due_count = sum(due)
            not_due_count = len(due) - due_count
            values[f"delayed_action_rate_at_{label}"] = (
                sum(
                    is_due and not is_predicted_due
                    for is_due, is_predicted_due in zip(
                        due, predicted_due, strict=True
                    )
                )
                / max(due_count, 1)
            )
            values[f"premature_action_rate_at_{label}"] = (
                sum(
                    not is_due and is_predicted_due
                    for is_due, is_predicted_due in zip(
                        due, predicted_due, strict=True
                    )
                )
                / max(not_due_count, 1)
            )
        return values

    def reconstruct_metrics(raw_predictions, original_targets):
        candidate_predictions = [
            min(max(value, 0.0), train_cap) for value in raw_predictions
        ]
        common_predictions = [
            min(max(value, 0.0), reporting_cap)
            for value in raw_predictions
        ]
        own_targets = [
            min(max(value, 0.0), train_cap) for value in original_targets
        ]
        reference_targets = [
            min(max(value, 0.0), reporting_cap)
            for value in original_targets
        ]
        return {
            "own_rmse": recompute_rmse(
                candidate_predictions, own_targets
            ),
            "reference_rmse": recompute_rmse(
                common_predictions, reference_targets
            ),
            "original_rmse": recompute_rmse(
                common_predictions, original_targets
            ),
            "nasa_score_reference": nasa_score(
                common_predictions, reference_targets
            ),
            "nasa_score_original": nasa_score(
                common_predictions, original_targets
            ),
            "saturation_fraction": sum(
                target > train_cap for target in original_targets
            )
            / len(original_targets),
            "exposure_bound": exposure(original_targets, train_cap),
            "joint_support_penalty": exposure(
                reference_targets, train_cap
            ),
            **decision_metrics(common_predictions, reference_targets),
        }

    def aggregate(values):
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values)
            / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "values": values,
        }

    def median(values):
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    def metrics_match(observed, expected):
        if set(expected) != set(OFFICIAL_SCALAR_METRICS):
            return False
        for metric, value in expected.items():
            candidate = observed.get(metric)
            if (
                candidate is None
                or isinstance(candidate, bool)
                or not close(candidate, value)
            ):
                return False
        return True

    for objective in ("self_scaled", "fixed_reference"):
        objective_payload = test_results.get(objective)
        if not isinstance(objective_payload, dict):
            return False
        rows = objective_payload.get("per_seed", [])
        if len(rows) != len(expected_seeds):
            return False
        scalar_values = {metric: [] for metric in OFFICIAL_SCALAR_METRICS}
        raw_by_seed = []
        common_by_seed = []
        candidate_by_seed = []
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "seed",
                    *OFFICIAL_ARRAY_FIELDS,
                    *OFFICIAL_SCALAR_METRICS,
                }
                or row.get("seed") != expected_seeds[index]
            ):
                return False
            units = row["units"]
            if (
                len(units) != expected_count
                or
                any(
                    isinstance(unit, bool)
                    or not isinstance(unit, int)
                    for unit in units
                )
                or len(set(units)) != len(units)
            ):
                return False
            numeric = {}
            for field in OFFICIAL_ARRAY_FIELDS[1:]:
                numeric[field] = numeric_list(row[field], expected_count)
                if numeric[field] is None:
                    return False
            raw = numeric["raw_predictions"]
            candidate = numeric["candidate_support_predictions"]
            common = numeric["common_support_predictions"]
            original = numeric["original_targets"]
            if any(
                not close(value, min(max(raw_value, 0.0), train_cap))
                for value, raw_value in zip(
                    candidate, raw, strict=True
                )
            ):
                return False
            if any(
                not close(
                    value,
                    min(max(raw_value, 0.0), reporting_cap),
                )
                for value, raw_value in zip(common, raw, strict=True)
            ):
                return False
            expected_metrics = reconstruct_metrics(raw, original)
            if not metrics_match(row, expected_metrics):
                return False
            for metric in OFFICIAL_SCALAR_METRICS:
                scalar_values[metric].append(expected_metrics[metric])
            raw_by_seed.append(raw)
            common_by_seed.append(common)
            candidate_by_seed.append(candidate)
            if baseline_units is None:
                baseline_units = units
                baseline_targets = original
            elif units != baseline_units or any(
                not close(left, right)
                for left, right in zip(
                    original, baseline_targets, strict=True
                )
            ):
                return False
        summary = objective_payload.get("summary")
        if (
            not isinstance(summary, dict)
            or set(summary) != set(OFFICIAL_SCALAR_METRICS)
        ):
            return False
        for metric in OFFICIAL_SCALAR_METRICS:
            observed = summary.get(metric)
            expected = aggregate(scalar_values[metric])
            if (
                not isinstance(observed, dict)
                or set(observed) != {"mean", "std", "values"}
            ):
                return False
            observed_values = numeric_list(
                observed.get("values"), len(expected_seeds)
            )
            if observed_values is None or any(
                not close(left, right)
                for left, right in zip(
                    observed_values, expected["values"], strict=True
                )
            ):
                return False
            if not close(observed.get("mean"), expected["mean"]) or not close(
                observed.get("std"), expected["std"]
            ):
                return False

        raw_ensemble = [
            median([values[index] for values in raw_by_seed])
            for index in range(expected_count)
        ]
        common_ensemble = [
            median([values[index] for values in common_by_seed])
            for index in range(expected_count)
        ]
        candidate_ensemble = [
            median([values[index] for values in candidate_by_seed])
            for index in range(expected_count)
        ]
        ensemble = objective_payload.get("ensemble")
        expected_ensemble_keys = {
            *OFFICIAL_ARRAY_FIELDS,
            *OFFICIAL_SCALAR_METRICS,
        }
        if (
            not isinstance(ensemble, dict)
            or set(ensemble) != expected_ensemble_keys
            or ensemble.get("units") != baseline_units
        ):
            return False
        ensemble_arrays = {}
        for field in OFFICIAL_ARRAY_FIELDS[1:]:
            ensemble_arrays[field] = numeric_list(
                ensemble.get(field), expected_count
            )
            if ensemble_arrays[field] is None:
                return False
        expected_arrays = {
            "raw_predictions": raw_ensemble,
            "candidate_support_predictions": candidate_ensemble,
            "common_support_predictions": common_ensemble,
            "original_targets": baseline_targets,
        }
        for field, expected_values in expected_arrays.items():
            if any(
                not close(left, right)
                for left, right in zip(
                    ensemble_arrays[field],
                    expected_values,
                    strict=True,
                )
            ):
                return False
        ensemble_metrics = reconstruct_metrics(raw_ensemble, baseline_targets)
        if not metrics_match(ensemble, ensemble_metrics):
            return False
    if stable_hash(
        {
            "units": baseline_units,
            "original_targets": baseline_targets,
        }
    ) != canonical_target_hash:
        return False
    return set(test_results) == {"self_scaled", "fixed_reference"}


def receipt_paths(output_root: Path, result_path: Path):
    base = output_root / "final_receipts"
    return (
        base / f"{result_path.name}.launch.json",
        base / f"{result_path.name}.receipt.json",
    )


def attempt_prefix(result_path: Path) -> str:
    return result_path.name.replace("/", "_").replace("\\", "_")


def attempt_integrity_payload(
    output_root: Path, result_path: Path
) -> dict[str, str]:
    """Return the immutable start/completion ledger for one planned result."""
    base = output_root / "final_attempts"
    if not base.exists():
        return {}
    prefix = attempt_prefix(result_path) + ".attempt"
    paths = sorted(
        (
            path
            for path in base.glob(prefix + "*.json")
            if path.is_file()
        ),
        key=lambda path: path.name,
    )
    starts = [
        path for path in paths if path.name.endswith(".start.json")
    ]
    expected_start_names = [
        f"{prefix}{index:03d}.start.json"
        for index in range(1, len(starts) + 1)
    ]
    if [path.name for path in starts] != expected_start_names:
        raise RuntimeError(
            f"Non-contiguous official-attempt ledger for {result_path}"
        )
    allowed = {path.name for path in starts}
    allowed.update(
        path.name.replace(".start.json", ".complete.json")
        for path in starts
    )
    if any(path.name not in allowed for path in paths):
        raise RuntimeError(
            f"Malformed official-attempt ledger for {result_path}"
        )
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verified_attempt_ledger(
    output_root: Path, result_path: Path
) -> tuple[dict[str, str], float]:
    """Validate attempt chronology and require a successful completion."""
    hashes = attempt_integrity_payload(output_root, result_path)
    if not hashes:
        raise RuntimeError(
            f"Official result has no execution-attempt ledger: {result_path}"
        )
    launch_path, _ = receipt_paths(output_root, result_path)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    authorized_unix = float(launch["authorized_unix"])
    if not math.isfinite(authorized_unix):
        raise RuntimeError("Launch authorization timestamp is invalid")
    starts = sorted(
        (
            Path(path)
            for path in hashes
            if path.endswith(".start.json")
        ),
        key=lambda path: path.name,
    )
    if len(starts) > MAX_OFFICIAL_ATTEMPTS:
        raise RuntimeError(
            f"Official result exceeds the frozen attempt limit: {result_path}"
        )
    previous_started = authorized_unix
    successful_completions = []
    incomplete_starts = []
    for index, start_path in enumerate(starts, start=1):
        start = json.loads(start_path.read_text(encoding="utf-8"))
        started_unix = float(start["started_unix"])
        expected_retry_reason = (
            "initial"
            if index == 1
            else "interrupted_attempt_without_completion_ledger"
        )
        if (
            start.get("protocol")
            != "confirmatory-final-attempt-start-v2"
            or start.get("attempt") != index
            or start.get("retry_reason") != expected_retry_reason
            or not math.isfinite(started_unix)
            or started_unix < previous_started
        ):
            raise RuntimeError(
                f"Official-attempt start chronology is invalid: {start_path}"
            )
        previous_started = started_unix
        complete_path = start_path.with_name(
            start_path.name.replace(".start.json", ".complete.json")
        )
        if not complete_path.exists():
            incomplete_starts.append(start_path)
            continue
        completion = json.loads(
            complete_path.read_text(encoding="utf-8")
        )
        completed_unix = float(completion["completed_unix"])
        returncode = completion.get("returncode")
        log_path = Path(start["log_path"])
        protocol = completion.get("protocol")
        common_invalid = (
            completion.get("attempt_start_sha256")
            != sha256_file(start_path)
            or not math.isfinite(completed_unix)
            or completed_unix < started_unix
            or not log_path.is_file()
            or completion.get("log_sha256") != sha256_file(log_path)
        )
        normal_invalid = (
            protocol == "confirmatory-final-attempt-complete-v2"
            and (
                isinstance(returncode, bool)
                or not isinstance(returncode, int)
            )
        )
        recovered_invalid = (
            protocol == "confirmatory-final-attempt-recovered-v2"
            and (
                completion.get("recovery_reason")
                != "valid_atomic_result_without_completion_ledger"
                or completion.get("result_sha256")
                != sha256_file(result_path)
            )
        )
        if (
            protocol
            not in {
                "confirmatory-final-attempt-complete-v2",
                "confirmatory-final-attempt-recovered-v2",
            }
            or common_invalid
            or normal_invalid
            or recovered_invalid
        ):
            raise RuntimeError(
                f"Official-attempt completion is invalid: {complete_path}"
            )
        if (
            protocol == "confirmatory-final-attempt-recovered-v2"
            or returncode == 0
        ):
            successful_completions.append(completed_unix)
        else:
            raise RuntimeError(
                f"Completed nonzero official attempt is terminal: "
                f"{complete_path}"
            )
    if len(incomplete_starts) > 1 or (
        incomplete_starts
        and incomplete_starts != starts[:-1]
    ):
        raise RuntimeError(
            f"Official attempt ledger has an invalid interrupted-attempt "
            f"pattern: {result_path}"
        )
    if len(successful_completions) != 1:
        raise RuntimeError(
            f"Official result must have exactly one successful completion: "
            f"{result_path}"
        )
    if not successful_completions:
        raise RuntimeError(
            f"Official result has no successful completed attempt: {result_path}"
        )
    return hashes, max(successful_completions)


def record_evaluation_attempt(
    output_root,
    result_path,
    spec,
    test_plan,
    started_path,
    command,
    log_path,
):
    """Create a non-overwritable record before a test process is launched."""
    launch_path, _ = receipt_paths(output_root, result_path)
    if not valid_launch_authorization(
        output_root, result_path, spec, test_plan, started_path
    ):
        raise RuntimeError("Cannot record an attempt without authorization")
    base = output_root / "final_attempts"
    base.mkdir(parents=True, exist_ok=True)
    existing = attempt_integrity_payload(output_root, result_path)
    starts = sorted(
        (
            Path(path)
            for path in existing
            if path.endswith(".start.json")
        ),
        key=lambda path: path.name,
    )
    start_count = len(starts)
    if start_count >= MAX_OFFICIAL_ATTEMPTS:
        raise RuntimeError(
            f"Official result reached the frozen attempt limit: {result_path}"
        )
    for start_path in starts:
        complete_path = start_path.with_name(
            start_path.name.replace(".start.json", ".complete.json")
        )
        if complete_path.exists():
            completion = json.loads(
                complete_path.read_text(encoding="utf-8")
            )
            if (
                completion.get("protocol")
                == "confirmatory-final-attempt-recovered-v2"
                or completion.get("returncode") == 0
            ):
                raise RuntimeError(
                    f"Official result already has a successful attempt: "
                    f"{result_path}"
                )
            raise RuntimeError(
                f"Completed nonzero official attempt is terminal: "
                f"{complete_path}"
            )
    if start_count > 1:
        raise RuntimeError(
            f"Official result has more than one interrupted attempt: "
            f"{result_path}"
        )
    index = start_count + 1
    prefix = attempt_prefix(result_path)
    start_path = base / f"{prefix}.attempt{index:03d}.start.json"
    complete_path = base / f"{prefix}.attempt{index:03d}.complete.json"
    value = {
        "protocol": "confirmatory-final-attempt-start-v2",
        "attempt": index,
        "retry_reason": (
            "initial"
            if index == 1
            else "interrupted_attempt_without_completion_ledger"
        ),
        "result_path": str(result_path.resolve()),
        "test_plan_sha256": stable_hash(test_plan),
        "start_marker_sha256": sha256_file(started_path),
        "launch_authorization_sha256": sha256_file(launch_path),
        "command": [str(value) for value in command],
        "command_sha256": stable_hash(
            {"command": [str(value) for value in command]}
        ),
        "log_path": str(log_path.resolve()),
        "started_unix": time.time(),
    }
    exclusive_json(start_path, value)
    return start_path, complete_path


def record_attempt_completion(start_path, complete_path, returncode, log_path):
    """Seal an attempt after the subprocess exits, including its full log."""
    if not start_path.is_file() or complete_path.exists():
        raise RuntimeError("Official-attempt completion state is invalid")
    if not log_path.is_file():
        raise RuntimeError("Official-attempt log is missing")
    exclusive_json(
        complete_path,
        {
            "protocol": "confirmatory-final-attempt-complete-v2",
            "attempt_start_sha256": sha256_file(start_path),
            "returncode": int(returncode),
            "log_sha256": sha256_file(log_path),
            "completed_unix": time.time(),
        },
    )


def recover_orphaned_atomic_result(
    output_root,
    result_path,
    spec,
    test_plan,
    started_path,
    result_is_valid,
):
    """Seal one interrupted attempt when its atomic result is fully valid."""
    if (
        not result_is_valid
        or not result_path.is_file()
        or not result_attestation_matches(
            output_root, result_path, spec, test_plan, started_path
        )
    ):
        return False
    hashes = attempt_integrity_payload(output_root, result_path)
    starts = sorted(
        (
            Path(path)
            for path in hashes
            if path.endswith(".start.json")
        ),
        key=lambda path: path.name,
    )
    if not starts:
        return False
    incomplete = [
        start
        for start in starts
        if not start.with_name(
            start.name.replace(".start.json", ".complete.json")
        ).exists()
    ]
    if len(incomplete) != 1 or incomplete[0] != starts[-1]:
        return False
    start_path = incomplete[0]
    start = json.loads(start_path.read_text(encoding="utf-8"))
    log_path = Path(start["log_path"])
    started_unix = float(start["started_unix"])
    if (
        not math.isfinite(started_unix)
        or not log_path.is_file()
        or result_path.stat().st_mtime < started_unix
        or start.get("test_plan_sha256") != stable_hash(test_plan)
        or start.get("start_marker_sha256") != sha256_file(started_path)
    ):
        return False
    complete_path = start_path.with_name(
        start_path.name.replace(".start.json", ".complete.json")
    )
    exclusive_json(
        complete_path,
        {
            "protocol": "confirmatory-final-attempt-recovered-v2",
            "attempt_start_sha256": sha256_file(start_path),
            "returncode": None,
            "log_sha256": sha256_file(log_path),
            "result_sha256": sha256_file(result_path),
            "recovery_reason": (
                "valid_atomic_result_without_completion_ledger"
            ),
            "completed_unix": time.time(),
        },
    )
    return True


def launch_core(result_path, spec, test_plan, started_path, nonce):
    return {
        "protocol": "confirmatory-final-launch-v2",
        "result_path": str(result_path.resolve()),
        "test_plan_sha256": stable_hash(test_plan),
        "start_marker_sha256": sha256_file(started_path),
        "expected_result": spec,
        "result_absent_before_authorization": True,
        "launch_nonce": nonce,
    }


def valid_launch_authorization(
    output_root, result_path, spec, test_plan, started_path
):
    launch_path, _ = receipt_paths(output_root, result_path)
    if not launch_path.exists():
        return False
    try:
        observed = json.loads(launch_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    nonce = observed.get("launch_nonce", "")
    if (
        len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        return False
    expected = launch_core(
        result_path, spec, test_plan, started_path, nonce
    )
    observed.pop("authorized_unix", None)
    return observed == expected


def ensure_launch_authorization(
    output_root, result_path, spec, test_plan, started_path
):
    if not started_path.exists():
        raise RuntimeError("Cannot authorize a result before the test marker")
    validate_planned_execution_sources(test_plan)
    if not valid_test_data_verification(
        output_root, test_plan, started_path
    ):
        raise RuntimeError(
            "Cannot authorize a result before plan-bound local data verification"
        )
    launch_path, receipt_path = receipt_paths(output_root, result_path)
    if receipt_path.exists():
        raise RuntimeError(
            f"Completion receipt exists before result launch: {receipt_path}"
        )
    if launch_path.exists():
        if not valid_launch_authorization(
            output_root, result_path, spec, test_plan, started_path
        ):
            raise RuntimeError(
                f"Launch authorization does not match test plan: {launch_path}"
            )
        return launch_path
    if result_path.exists():
        raise RuntimeError(
            f"Result existed before launch authorization: {result_path}"
        )
    value = launch_core(
        result_path,
        spec,
        test_plan,
        started_path,
        secrets.token_hex(32),
    )
    value["authorized_unix"] = time.time()
    exclusive_json(launch_path, value)
    return launch_path


def execution_environment(
    output_root, result_path, spec, test_plan, started_path
):
    validate_planned_execution_sources(test_plan)
    if not valid_test_data_verification(
        output_root, test_plan, started_path
    ):
        raise RuntimeError("Official test-data verification receipt is invalid")
    launch_path, _ = receipt_paths(output_root, result_path)
    if not valid_launch_authorization(
        output_root, result_path, spec, test_plan, started_path
    ):
        raise RuntimeError("Cannot build environment from invalid launch")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    environment = dict(os.environ)
    environment.update(
        {
            "CCV2_TEST_PLAN_SHA256": stable_hash(test_plan),
            "CCV2_START_MARKER_SHA256": sha256_file(started_path),
            "CCV2_LAUNCH_SHA256": sha256_file(launch_path),
            "CCV2_LAUNCH_NONCE": launch["launch_nonce"],
        }
    )
    return environment


def result_attestation_matches(
    output_root, result_path, spec, test_plan, started_path
):
    launch_path, _ = receipt_paths(output_root, result_path)
    if not valid_launch_authorization(
        output_root, result_path, spec, test_plan, started_path
    ):
        return False
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        observed = result["official_execution_attestation"]
    except Exception:
        return False
    expected = {
        "protocol": "confirmatory-final-execution-attestation-v2",
        "test_plan_sha256": stable_hash(test_plan),
        "start_marker_sha256": sha256_file(started_path),
        "launch_authorization_sha256": sha256_file(launch_path),
        "launch_nonce": launch["launch_nonce"],
    }
    return observed == expected


def receipt_core(
    output_root, result_path, spec, test_plan, started_path
):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    launch_path, _ = receipt_paths(output_root, result_path)
    if not result_attestation_matches(
        output_root, result_path, spec, test_plan, started_path
    ):
        raise RuntimeError("Result execution attestation is invalid")
    attempt_hashes, _ = verified_attempt_ledger(
        output_root, result_path
    )
    return {
        "protocol": "confirmatory-final-result-receipt-v2",
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "test_plan_sha256": stable_hash(test_plan),
        "start_marker_sha256": sha256_file(started_path),
        "launch_authorization_sha256": sha256_file(launch_path),
        "expected_result": spec,
        "test_data_fingerprints": {
            "test_sha256": result["data_fingerprints"]["test_sha256"],
            "truth_sha256": result["data_fingerprints"]["truth_sha256"],
        },
        "attempt_file_sha256": attempt_hashes,
    }


def validate_result_receipt(
    output_root, result_path, spec, test_plan, started_path
):
    launch_path, receipt_path = receipt_paths(output_root, result_path)
    if (
        not result_path.exists()
        or not receipt_path.exists()
        or not valid_launch_authorization(
            output_root, result_path, spec, test_plan, started_path
        )
        or not result_attestation_matches(
            output_root, result_path, spec, test_plan, started_path
        )
    ):
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        canonical = test_plan["canonical_test_data"][spec["dataset"]]
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_completed_unix = float(observed["completed_unix"])
        _, latest_successful_completion = verified_attempt_ledger(
            output_root, result_path
        )
        expected = receipt_core(
            output_root, result_path, spec, test_plan, started_path
        )
    except Exception:
        return False
    if (
        not result_matches_plan_spec(result, spec, canonical)
        or
        not math.isfinite(receipt_completed_unix)
        or receipt_completed_unix < latest_successful_completion
    ):
        return False
    observed.pop("completed_unix", None)
    return observed == expected


def write_result_receipt(
    output_root, result_path, spec, test_plan, started_path
):
    if not valid_launch_authorization(
        output_root, result_path, spec, test_plan, started_path
    ):
        raise RuntimeError("Result has no matching launch authorization")
    _, receipt_path = receipt_paths(output_root, result_path)
    value = receipt_core(
        output_root, result_path, spec, test_plan, started_path
    )
    value["completed_unix"] = time.time()
    if receipt_path.exists():
        if not validate_result_receipt(
            output_root, result_path, spec, test_plan, started_path
        ):
            raise RuntimeError(
                f"Existing completion receipt is invalid: {receipt_path}"
            )
    else:
        exclusive_json(receipt_path, value)
    return receipt_path


def recover_receipt(
    output_root,
    result_path,
    spec,
    test_plan,
    started_path,
    result_is_valid,
):
    if not result_is_valid:
        return False
    if not valid_launch_authorization(
        output_root, result_path, spec, test_plan, started_path
    ):
        return False
    try:
        verified_attempt_ledger(output_root, result_path)
    except RuntimeError:
        if not recover_orphaned_atomic_result(
            output_root,
            result_path,
            spec,
            test_plan,
            started_path,
            result_is_valid,
        ):
            return False
    write_result_receipt(
        output_root, result_path, spec, test_plan, started_path
    )
    return validate_result_receipt(
        output_root, result_path, spec, test_plan, started_path
    )


def validate_final(
    result_path: Path,
    manifest: dict,
    dataset: str,
    cap: int,
    args,
    canonical,
):
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = expected_config(manifest, dataset)
    return (
        result.get("protocol") == "confirmatory-rul-cap-v2"
        and result.get("campaign_id") == manifest["campaign_id"]
        and result.get("dataset") == dataset
        and int(result.get("train_cap")) == int(cap)
        and float(result.get("reporting_cap")) == 125.0
        and result.get("validation_only") is False
        and result.get("test_evaluated") is True
        and int(result.get("split_seed")) == args.primary_split_seed
        and int(result.get("endpoint_seed"))
        == args.primary_split_seed + 1000003
        and result.get("model_seeds")
        == list(range(args.seed_base, args.seed_base + args.final_seeds))
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == manifest["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == manifest["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == manifest["raw_train_sha256"][dataset]
        and result_data_matches(result, canonical)
        and per_seed_prediction_payload_valid(result, canonical)
        and config_matches(result.get("config"), expected)
    )


def run_final(
    args,
    manifest,
    dataset,
    cap,
    test_plan,
    started_path,
):
    tag = (
        f"locked_cap{cap}_split{args.primary_split_seed}_"
        f"seeds{args.final_seeds}"
    )
    result_path = (
        args.output_root / "results" / f"{dataset}_{tag}.json"
    )
    canonical = test_plan["canonical_test_data"][dataset]
    spec = test_plan["cap_result_specs"][dataset][str(cap)]
    if validate_final(
        result_path, manifest, dataset, cap, args, canonical
    ) and validate_result_receipt(
        args.output_root, result_path, spec, test_plan, started_path
    ):
        print(f"SKIP immutable final {dataset} cap={cap}", flush=True)
        return result_path
    if result_path.exists():
        try:
            existing_result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        except Exception:
            existing_result = {}
        result_is_valid = (
            validate_final(
                result_path, manifest, dataset, cap, args, canonical
            )
            and result_matches_plan_spec(
                existing_result, spec, canonical
            )
        )
        if recover_receipt(
            args.output_root,
            result_path,
            spec,
            test_plan,
            started_path,
            result_is_valid,
        ):
            print(
                f"RECOVER receipt for immutable final {dataset} cap={cap}",
                flush=True,
            )
            return result_path
        raise RuntimeError(
            f"Final result exists without a valid plan-bound receipt: "
            f"{result_path}"
        )
    ensure_launch_authorization(
        args.output_root, result_path, spec, test_plan, started_path
    )
    command = [
        sys.executable,
        "-u",
        str(FINAL_ATTESTED_TRAINER),
        "--dataset",
        dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--campaign_id",
        manifest["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(cap),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(args.primary_split_seed),
        "--endpoint_seed",
        str(args.primary_split_seed + 1000003),
        "--seeds",
        str(args.final_seeds),
        "--seed_base",
        str(args.seed_base),
        "--epochs",
        str(manifest["epochs"]),
        "--patience",
        str(manifest["patience"]),
        "--train_stride",
        str(manifest["train_stride"]),
        "--validation_interval",
        str(manifest["validation_interval"]),
        "--carrier",
        manifest["carrier"],
    ]
    log_path = (
        args.output_root
        / "final_logs"
        / f"{dataset}_{tag}_{manifest['campaign_id']}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        attempt = 2
        while True:
            candidate = log_path.with_name(
                f"{log_path.stem}_attempt{attempt}{log_path.suffix}"
            )
            if not candidate.exists():
                log_path = candidate
                break
            attempt += 1
    print(f"RUN immutable final {dataset} cap={cap}", flush=True)
    environment = execution_environment(
        args.output_root, result_path, spec, test_plan, started_path
    )
    attempt_start, attempt_complete = record_evaluation_attempt(
        args.output_root,
        result_path,
        spec,
        test_plan,
        started_path,
        command,
        log_path,
    )
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    record_attempt_completion(
        attempt_start, attempt_complete, process.returncode, log_path
    )
    if process.returncode != 0:
        raise RuntimeError(f"Final run failed; see {log_path}")
    if not validate_final(
        result_path, manifest, dataset, cap, args, canonical
    ):
        raise RuntimeError(
            f"Final result failed manifest validation: {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result_matches_plan_spec(result, spec, canonical):
        raise RuntimeError(
            f"Final result differs from immutable plan: {result_path}"
        )
    write_result_receipt(
        args.output_root, result_path, spec, test_plan, started_path
    )
    return result_path


def validate_hpo_final(
    result_path,
    screening,
    tpe_lock,
    dataset,
    params,
    args,
    canonical,
):
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result["config"]
    except Exception:
        return False
    expected = expected_config(
        screening,
        dataset,
        {
            "lr": params["lr"],
            "batch": params["batch"],
            "huber_delta": params["huber_delta"],
        },
    )
    return (
        result.get("protocol") == "confirmatory-rul-cap-v2"
        and
        result.get("campaign_id") == tpe_lock["campaign_id"]
        and result.get("dataset") == dataset
        and result.get("validation_only") is False
        and result.get("test_evaluated") is True
        and float(result.get("reporting_cap")) == 125.0
        and result.get("split_seed") == args.primary_split_seed
        and int(result.get("endpoint_seed"))
        == args.primary_split_seed + 1000003
        and result.get("model_seeds")
        == list(range(args.seed_base, args.seed_base + args.final_seeds))
        and int(result.get("train_cap")) == int(params["train_cap"])
        and config_matches(config, expected)
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == screening["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == screening["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == screening["raw_train_sha256"][dataset]
        and result_data_matches(result, canonical)
        and per_seed_prediction_payload_valid(result, canonical)
    )


def run_hpo_final(
    args,
    screening,
    tpe_lock,
    dataset,
    objective,
    test_plan,
    started_path,
):
    params = tpe_lock["best_params"]
    tag = (
        f"locked_tpe_{objective}_trial{tpe_lock['best_trial']:03d}_"
        f"seeds{args.final_seeds}"
    )
    result_path = (
        args.output_root / "results" / f"{dataset}_{tag}.json"
    )
    canonical = test_plan["canonical_test_data"][dataset]
    spec = test_plan["tpe_result_specs"][f"{dataset}_{objective}"]
    if validate_hpo_final(
        result_path,
        screening,
        tpe_lock,
        dataset,
        params,
        args,
        canonical,
    ) and validate_result_receipt(
        args.output_root, result_path, spec, test_plan, started_path
    ):
        print(f"SKIP immutable TPE final {dataset} {objective}", flush=True)
        return result_path
    if result_path.exists():
        try:
            existing_result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        except Exception:
            existing_result = {}
        result_is_valid = (
            validate_hpo_final(
                result_path,
                screening,
                tpe_lock,
                dataset,
                params,
                args,
                canonical,
            )
            and result_matches_plan_spec(
                existing_result, spec, canonical
            )
        )
        if recover_receipt(
            args.output_root,
            result_path,
            spec,
            test_plan,
            started_path,
            result_is_valid,
        ):
            print(
                f"RECOVER receipt for immutable TPE final "
                f"{dataset} {objective}",
                flush=True,
            )
            return result_path
        raise RuntimeError(
            f"Immutable TPE final result lacks a valid plan-bound receipt: "
            f"{result_path}"
        )
    ensure_launch_authorization(
        args.output_root, result_path, spec, test_plan, started_path
    )
    command = [
        sys.executable,
        "-u",
        str(FINAL_ATTESTED_TRAINER),
        "--dataset",
        dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--campaign_id",
        tpe_lock["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(params["train_cap"]),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(args.primary_split_seed),
        "--endpoint_seed",
        str(args.primary_split_seed + 1000003),
        "--seeds",
        str(args.final_seeds),
        "--seed_base",
        str(args.seed_base),
        "--seq_len",
        str(DEFAULT_SEQ[dataset]),
        "--lr",
        str(params["lr"]),
        "--batch",
        str(params["batch"]),
        "--huber_delta",
        str(params["huber_delta"]),
        "--epochs",
        str(screening["epochs"]),
        "--patience",
        str(screening["patience"]),
        "--train_stride",
        str(screening["train_stride"]),
        "--validation_interval",
        str(screening["validation_interval"]),
        "--carrier",
        screening["carrier"],
    ]
    log = (
        args.output_root
        / "final_logs"
        / f"{dataset}_{tag}_{tpe_lock['campaign_id']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        attempt = 2
        while True:
            candidate = log.with_name(
                f"{log.stem}_attempt{attempt}{log.suffix}"
            )
            if not candidate.exists():
                log = candidate
                break
            attempt += 1
    print(f"RUN immutable TPE final {dataset} {objective}", flush=True)
    environment = execution_environment(
        args.output_root, result_path, spec, test_plan, started_path
    )
    attempt_start, attempt_complete = record_evaluation_attempt(
        args.output_root,
        result_path,
        spec,
        test_plan,
        started_path,
        command,
        log,
    )
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    record_attempt_completion(
        attempt_start, attempt_complete, process.returncode, log
    )
    if process.returncode != 0:
        raise RuntimeError(f"TPE final run failed; see {log}")
    if not validate_hpo_final(
        result_path,
        screening,
        tpe_lock,
        dataset,
        params,
        args,
        canonical,
    ):
        raise RuntimeError(
            f"TPE final result failed validation: {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result_matches_plan_spec(result, spec, canonical):
        raise RuntimeError(
            f"TPE final result differs from immutable plan: {result_path}"
        )
    write_result_receipt(
        args.output_root, result_path, spec, test_plan, started_path
    )
    return result_path


def flatten_result_paths(result_paths, tpe_result_paths):
    paths = []
    for caps in result_paths.values():
        paths.extend(Path(value) for value in caps.values())
    paths.extend(Path(value) for value in tpe_result_paths.values())
    return sorted(set(paths), key=str)


def planned_result_paths(output_root, test_plan):
    names = []
    for dataset_specs in test_plan["cap_result_specs"].values():
        names.extend(
            spec["result_filename"] for spec in dataset_specs.values()
        )
    names.extend(
        spec["result_filename"]
        for spec in test_plan["tpe_result_specs"].values()
    )
    if len(names) != len(set(names)):
        raise RuntimeError("Immutable test plan contains duplicate results")
    return {
        (output_root / "results" / name).resolve() for name in names
    }


def result_integrity_payload(result_paths, tpe_result_paths):
    hashes = {}
    data_fingerprints = {}
    for path in flatten_result_paths(result_paths, tpe_result_paths):
        if not path.exists():
            raise RuntimeError(f"Missing final result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("protocol") != "confirmatory-rul-cap-v2"
            or result.get("validation_only") is not False
            or result.get("test_evaluated") is not True
        ):
            raise RuntimeError(f"Invalid final result state: {path}")
        hashes[str(path)] = sha256_file(path)
        dataset = result["dataset"]
        pair = {
            "test_sha256": result["data_fingerprints"]["test_sha256"],
            "truth_sha256": result["data_fingerprints"]["truth_sha256"],
        }
        if dataset in data_fingerprints:
            if data_fingerprints[dataset] != pair:
                raise RuntimeError(
                    f"Test/truth fingerprints differ within {dataset}"
                )
        else:
            data_fingerprints[dataset] = pair
    return hashes, data_fingerprints


def receipt_integrity_payload(
    result_paths,
    tpe_result_paths,
    output_root,
    test_plan,
    started_path,
):
    specs = {}
    for dataset_specs in test_plan["cap_result_specs"].values():
        for spec in dataset_specs.values():
            specs[spec["result_filename"]] = spec
    for spec in test_plan["tpe_result_specs"].values():
        specs[spec["result_filename"]] = spec
    observed_paths = {
        path.resolve()
        for path in flatten_result_paths(
            result_paths, tpe_result_paths
        )
    }
    expected_paths = planned_result_paths(output_root, test_plan)
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        extra = sorted(str(path) for path in observed_paths - expected_paths)
        raise RuntimeError(
            "Final result set differs from immutable plan; "
            f"missing={missing}, extra={extra}"
        )
    hashes = {}
    for path in sorted(observed_paths, key=str):
        spec = specs.get(path.name)
        if spec is None:
            raise RuntimeError(
                f"Final result is absent from immutable plan: {path}"
            )
        if not validate_result_receipt(
            output_root, path, spec, test_plan, started_path
        ):
            raise RuntimeError(
                f"Invalid plan-bound completion receipt for {path}"
            )
        _, receipt_path = receipt_paths(output_root, path)
        hashes[str(receipt_path)] = sha256_file(receipt_path)
    return hashes


def validate_summary_chain(
    summary,
    test_plan,
    output_root,
    started_path,
):
    """Validate a completed summary independently of its final marker."""
    data_receipt_path = test_data_verification_path(output_root)
    if (
        summary.get("protocol") != "confirmatory-final-summary-v2"
        or summary.get("test_plan_sha256") != stable_hash(test_plan)
        or summary.get("start_marker_sha256")
        != sha256_file(started_path)
        or not valid_test_data_verification(
            output_root, test_plan, started_path
        )
        or summary.get("test_data_verification_receipt_sha256")
        != sha256_file(data_receipt_path)
    ):
        raise RuntimeError("Final-summary hash chain is invalid")
    observed_hashes, observed_data = result_integrity_payload(
        summary["result_paths"],
        summary["tpe_result_paths"],
    )
    if (
        observed_hashes != summary.get("result_file_sha256")
        or observed_data != summary.get("test_data_fingerprints")
    ):
        raise RuntimeError("Completed final results changed after summary")
    specs = {}
    for dataset_specs in test_plan["cap_result_specs"].values():
        for spec in dataset_specs.values():
            specs[spec["result_filename"]] = spec
    for spec in test_plan["tpe_result_specs"].values():
        specs[spec["result_filename"]] = spec
    for path in flatten_result_paths(
        summary["result_paths"], summary["tpe_result_paths"]
    ):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            spec = specs[path.name]
            canonical = test_plan["canonical_test_data"][result["dataset"]]
        except Exception as exc:
            raise RuntimeError(
                f"Cannot semantically revalidate completed result: {path}"
            ) from exc
        if not result_matches_plan_spec(result, spec, canonical):
            raise RuntimeError(
                f"Completed result fails semantic reconstruction: {path}"
            )
    observed_receipts = receipt_integrity_payload(
        summary["result_paths"],
        summary["tpe_result_paths"],
        output_root,
        test_plan,
        started_path,
    )
    if observed_receipts != summary.get("receipt_file_sha256"):
        raise RuntimeError("Final result receipts changed after summary")


def completion_marker_payload(summary: dict) -> dict:
    return {
        "protocol": "confirmatory-final-completion-marker-v2",
        "campaign_id": summary["campaign_id"],
        "final_summary_sha256": stable_hash(summary),
        "completed_unix": summary["completed_unix"],
    }


def validate_completed_chain(
    summary,
    marker,
    test_plan,
    output_root,
    started_path,
):
    validate_summary_chain(
        summary, test_plan, output_root, started_path
    )
    expected = completion_marker_payload(summary)
    if marker != expected:
        raise RuntimeError("Completed-test marker is invalid")


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    acquire_single_instance_lock(
        args.output_root / "OFFICIAL_TEST_WRAPPER.lock"
    )
    if args.final_seeds != 5 or args.seed_base != 45:
        raise RuntimeError(
            "The official protocol is locked to model seeds 45--49, "
            "which are disjoint from selection seeds 42--44"
        )
    manifest_path = args.output_root / "campaign_manifest.json"
    lock_path = args.output_root / "final_cap_decisions.json"
    if not manifest_path.exists() or not lock_path.exists():
        raise RuntimeError("Validation campaign is not complete and locked")
    preflight = run_preflight(args.output_root, args.data_root)
    if (
        preflight.get("status") != "pass"
        or preflight.get("test_or_truth_files_opened") is not False
    ):
        raise RuntimeError("Pre-test integrity audit did not pass")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_paths = {
        "selection_attempt_manifest": (
            args.output_root / "SELECTION_ATTEMPT_MANIFEST.json"
        ),
        "qa_manifest": (
            args.output_root / "qa" / "pytest_run_manifest.json"
        ),
        "qa_junit": args.output_root / "qa" / "pytest-junit.xml",
        "premarker_campaign_inventory": (
            args.output_root / "PREMARKER_CAMPAIGN_INVENTORY.json"
        ),
    }
    for label, path in provenance_paths.items():
        if not path.is_file():
            raise RuntimeError(
                f"Pre-test provenance artefact is missing: {label}"
            )
    selection_attempt_manifest = json.loads(
        provenance_paths["selection_attempt_manifest"].read_text(
            encoding="utf-8"
        )
    )
    qa_manifest = json.loads(
        provenance_paths["qa_manifest"].read_text(encoding="utf-8")
    )
    premarker_inventory = json.loads(
        provenance_paths["premarker_campaign_inventory"].read_text(
            encoding="utf-8"
        )
    )
    expected_preflight_provenance = {
        "selection_attempt_manifest_sha256": stable_hash(
            selection_attempt_manifest
        ),
        "qa_manifest_sha256": stable_hash(qa_manifest),
        "qa_junit_sha256": sha256_file(
            provenance_paths["qa_junit"]
        ),
        "premarker_campaign_inventory_sha256": stable_hash(
            premarker_inventory
        ),
    }
    for key, expected in expected_preflight_provenance.items():
        if preflight.get(key) != expected:
            raise RuntimeError(
                f"Preflight provenance binding differs: {key}"
            )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(TRAINER) != manifest.get("trainer_sha256"):
        raise RuntimeError("Trainer source changed after screening")
    if not FINAL_ATTESTED_TRAINER.is_file():
        raise RuntimeError("Official execution-attestation shim is missing")
    if args.primary_split_seed != manifest["split_seeds"][0]:
        raise RuntimeError("Official split differs from the campaign manifest")
    if manifest["campaign_id"] != lock.get("screening_campaign_id"):
        raise RuntimeError("Screening campaign and final decision lock differ")
    stability_path = args.output_root / "stability_manifest.json"
    if not stability_path.exists():
        raise RuntimeError("Model-seed stability manifest is missing")
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability.get("campaign_id") != lock.get("stability_campaign_id"):
        raise RuntimeError("Stability campaign and final decision lock differ")
    if stable_hash(stability) != lock.get("stability_manifest_sha256"):
        raise RuntimeError("Stability manifest changed after locking")
    if lock.get("test_evaluated") is not False:
        raise RuntimeError("Decision lock is not in the pre-test state")
    robustness_path = args.output_root / "tpe_robustness_summary.json"
    if not robustness_path.exists():
        raise RuntimeError("Frozen TPE robustness checks are incomplete")
    robustness = json.loads(
        robustness_path.read_text(encoding="utf-8")
    )
    if (
        robustness.get("status") != "complete"
        or robustness.get("selection_frozen") is not True
        or robustness.get("test_evaluated") is not False
    ):
        raise RuntimeError("TPE robustness summary is not pre-test complete")
    carrier_transfer_path = (
        args.output_root / "carrier_transfer_summary.json"
    )
    if not carrier_transfer_path.exists():
        raise RuntimeError("Second-carrier transfer check is incomplete")
    carrier_transfer = json.loads(
        carrier_transfer_path.read_text(encoding="utf-8")
    )
    if not carrier_transfer_gate_passes(carrier_transfer):
        raise RuntimeError(
            "Second-carrier transfer summary is neither pre-test complete "
            "nor an audited, frozen terminal failure"
        )
    tpe_locks = {}
    for dataset, objective in TPE_STUDIES:
        lock_file = (
            args.output_root
            / "tpe_locks"
            / f"{dataset}_{objective}.json"
        )
        manifest_file = (
            args.output_root
            / "tpe_manifests"
            / f"{dataset}_{objective}.json"
        )
        if not lock_file.exists() or not manifest_file.exists():
            raise RuntimeError(
                f"Missing TPE lock or manifest for {dataset} {objective}"
            )
        value = json.loads(lock_file.read_text(encoding="utf-8"))
        tpe_manifest = json.loads(
            manifest_file.read_text(encoding="utf-8")
        )
        if value.get("test_evaluated") is not False:
            raise RuntimeError(f"TPE lock has test state: {lock_file}")
        if (
            value.get("campaign_id") != tpe_manifest.get("campaign_id")
            or value.get("manifest_sha256") != stable_hash(tpe_manifest)
            or tpe_manifest.get("screening_campaign_id")
            != manifest["campaign_id"]
            or tpe_manifest.get("trainer_sha256")
            != manifest["trainer_sha256"]
            or int(value.get("complete_trials", 0))
            != int(tpe_manifest.get("trials", -1))
        ):
            raise RuntimeError(
                f"TPE lock does not match its manifest: {lock_file}"
            )
        tpe_locks[(dataset, objective)] = value

    requested_caps = {}
    for dataset in DATASETS:
        decisions = lock["decisions"][dataset]
        requested_caps[dataset] = sorted(
            {
                125,
                int(decisions["self_scaled"]["selected_cap"]),
                int(decisions["fixed_reference"]["selected_cap"]),
            }
        )
    canonical_data, canonical_source = trusted_test_fingerprints()
    model_seeds = list(
        range(args.seed_base, args.seed_base + args.final_seeds)
    )
    runtime_environment = capture_runtime_environment()
    cap_result_specs = {}
    for dataset in DATASETS:
        cap_result_specs[dataset] = {}
        for cap in requested_caps[dataset]:
            tag = (
                f"locked_cap{cap}_split{args.primary_split_seed}_"
                f"seeds{args.final_seeds}"
            )
            cap_result_specs[dataset][str(cap)] = {
                "result_filename": f"{dataset}_{tag}.json",
                "campaign_id": manifest["campaign_id"],
                "dataset": dataset,
                "tag": tag,
                "train_cap": int(cap),
                "reporting_cap": REPORTING_CAP,
                "split_seed": args.primary_split_seed,
                "endpoint_seed": args.primary_split_seed + 1000003,
                "model_seeds": model_seeds,
                "config": expected_config(manifest, dataset),
                "trainer_sha256": manifest["trainer_sha256"],
                "model_sha256": manifest["model_sha256"],
                "train_sha256": manifest["raw_train_sha256"][dataset],
                "runtime_environment": runtime_environment,
            }
    tpe_result_specs = {}
    for dataset, objective in TPE_STUDIES:
        tpe_lock = tpe_locks[(dataset, objective)]
        params = tpe_lock["best_params"]
        tag = (
            f"locked_tpe_{objective}_trial{tpe_lock['best_trial']:03d}_"
            f"seeds{args.final_seeds}"
        )
        tpe_result_specs[f"{dataset}_{objective}"] = {
            "result_filename": f"{dataset}_{tag}.json",
            "campaign_id": tpe_lock["campaign_id"],
            "dataset": dataset,
            "tag": tag,
            "objective": objective,
            "train_cap": int(params["train_cap"]),
            "reporting_cap": REPORTING_CAP,
            "split_seed": args.primary_split_seed,
            "endpoint_seed": args.primary_split_seed + 1000003,
            "model_seeds": model_seeds,
            "config": expected_config(
                manifest,
                dataset,
                {
                    "lr": params["lr"],
                    "batch": params["batch"],
                    "huber_delta": params["huber_delta"],
                },
            ),
            "trainer_sha256": manifest["trainer_sha256"],
            "model_sha256": manifest["model_sha256"],
            "train_sha256": manifest["raw_train_sha256"][dataset],
            "runtime_environment": runtime_environment,
        }
    test_plan = {
        "protocol": "confirmatory-test-plan-v2",
        "campaign_id": manifest["campaign_id"],
        "decision_lock_sha256": stable_hash(lock),
        "primary_split_seed": args.primary_split_seed,
        "model_seeds": model_seeds,
        "requested_caps": requested_caps,
        "cap_result_specs": cap_result_specs,
        "tpe_result_specs": tpe_result_specs,
        "canonical_test_data": canonical_data,
        "canonical_source": canonical_source,
        "runtime_environment": runtime_environment,
        "environment_lock_sha256": sha256_file(ENVIRONMENT_LOCK),
        "tpe_locks": {
            f"{dataset}_{objective}": {
                "sha256": stable_hash(tpe_locks[(dataset, objective)]),
                "best_trial": tpe_locks[(dataset, objective)]["best_trial"],
                "best_params": tpe_locks[(dataset, objective)]["best_params"],
            }
            for dataset, objective in TPE_STUDIES
        },
        "tpe_robustness_summary_sha256": stable_hash(robustness),
        "carrier_transfer_summary_sha256": stable_hash(carrier_transfer),
        **expected_preflight_provenance,
        "preflight_report_sha256": stable_hash(preflight),
        **planned_execution_source_hashes(manifest),
        "analysis_plan": {
            "primary_family": {
                "comparison": (
                    "cap_grid_fixed_reference_minus_"
                    "cap_grid_self_scaled"
                ),
                "datasets": list(DATASETS),
                "metric": "reference_scale_rmse_common_support",
                "paired_unit": "model_seed",
                "multiplicity": "holm_within_four_contrasts",
            },
            "secondary_family": {
                "comparisons": [
                    "cap_grid_fixed_reference_minus_fixed_cap_125",
                    "tpe_fixed_reference_minus_tpe_self_scaled",
                    "tpe_fixed_reference_minus_tpe_fixed_cap_125",
                ],
                "multiplicity": "holm_within_eight_contrasts",
            },
            "all_contrasts_sensitivity": (
                "holm_across_all_twelve_contrasts"
            ),
            "ensemble_uncertainty": (
                "paired_engine_percentile_bootstrap_10000"
            ),
            "individual_method_uncertainty": (
                "engine_percentile_bootstrap_2000"
            ),
            "seed_test": (
                "two_sided_exact_sign_flip_wilcoxon_zero_dropped_"
                "average_tied_ranks"
            ),
            "seed_test_zero_tolerance": 1e-12,
            "official_attempt_policy": {
                "maximum_attempts_per_result": MAX_OFFICIAL_ATTEMPTS,
                "retry_policy": OFFICIAL_RETRY_POLICY,
            },
            "maintenance_proxy_thresholds": [20, 30, 50],
            "reference_scale_nasa_score": "descriptive",
            "target_support_rescore": (
                "four_cells_plus_mse_scale_interaction_descriptive"
            ),
            "inference_scope": (
                "descriptive_conditional_summaries_due_to_disclosed_"
                "prior_cmapss_test_contact"
            ),
            "pooled_independent_fleet_claim": False,
            "binary_success_threshold": None,
        },
        "created_unix": time.time(),
        "selection_frozen_before_test": True,
        "same_plan_resumption_allowed": True,
    }
    test_plan_path = args.output_root / "immutable_test_plan.json"
    if test_plan_path.exists():
        existing = json.loads(test_plan_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_plan = dict(test_plan)
        comparable_existing.pop("created_unix", None)
        comparable_plan.pop("created_unix", None)
        if comparable_existing != comparable_plan:
            raise RuntimeError("Existing immutable test plan differs")
        test_plan = existing
    else:
        exclusive_json(test_plan_path, test_plan)
    validate_planned_execution_sources(test_plan)

    started_path = args.output_root / "TEST_EVALUATION_STARTED.json"
    summary_path = args.output_root / "final_test_summary.json"
    completed_marker_path = args.output_root / "TEST_EVALUATED"
    if started_path.exists():
        if summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("campaign_id") != manifest["campaign_id"]:
                raise RuntimeError("Completed test summary belongs to another campaign")
            if not completed_marker_path.exists():
                validate_summary_chain(
                    existing,
                    json.loads(
                        test_plan_path.read_text(encoding="utf-8")
                    ),
                    args.output_root,
                    started_path,
                )
                exclusive_json(
                    completed_marker_path,
                    completion_marker_payload(existing),
                )
                print(
                    "RECOVER completion marker from validated immutable "
                    "final summary",
                    flush=True,
                )
            completed_marker = json.loads(
                completed_marker_path.read_text(encoding="utf-8")
            )
            validate_completed_chain(
                existing,
                completed_marker,
                json.loads(
                    test_plan_path.read_text(encoding="utf-8")
                ),
                args.output_root,
                started_path,
            )
            print(json.dumps(existing, indent=2), flush=True)
            return
        started = json.loads(started_path.read_text(encoding="utf-8"))
        expected_plan_hash = stable_hash(
            json.loads(test_plan_path.read_text(encoding="utf-8"))
        )
        if (
            started.get("campaign_id") != manifest["campaign_id"]
            or started.get("test_plan_sha256") != expected_plan_hash
            or started.get("canonical_test_data")
            != test_plan["canonical_test_data"]
            or started.get("canonical_source")
            != test_plan["canonical_source"]
            or started.get("selection_changes_forbidden") is not True
        ):
            raise RuntimeError(
                "Existing test-start marker does not match the immutable plan"
            )
        print(
            "RESUME the same immutable test plan; completed results will be "
            "validated and skipped",
            flush=True,
        )
    else:
        current_inventory = build_premarker_inventory(
            args.output_root,
            manifest["campaign_id"],
            allow_immutable_plan=True,
        )
        if current_inventory != premarker_inventory:
            raise RuntimeError(
                "Campaign JSON inventory changed after preflight and before "
                "the immutable test-start marker"
            )
        expected_result_paths = []
        for dataset_specs in test_plan["cap_result_specs"].values():
            for spec in dataset_specs.values():
                expected_result_paths.append(
                    args.output_root
                    / "results"
                    / spec["result_filename"]
                )
        for spec in test_plan["tpe_result_specs"].values():
            expected_result_paths.append(
                args.output_root / "results" / spec["result_filename"]
            )
        for result_path in expected_result_paths:
            launch_path, receipt_path = receipt_paths(
                args.output_root, result_path
            )
            for forbidden in (result_path, launch_path, receipt_path):
                if forbidden.exists():
                    raise RuntimeError(
                        "A planned test artifact predates the immutable "
                        f"test-start marker: {forbidden}"
                    )
        if summary_path.exists() or completed_marker_path.exists():
            raise RuntimeError(
                "Completion artifacts exist before the test-start marker"
            )
        exclusive_json(
            started_path,
            {
                "campaign_id": manifest["campaign_id"],
                "test_plan_sha256": stable_hash(
                    json.loads(
                        test_plan_path.read_text(encoding="utf-8")
                    )
                ),
                "canonical_test_data": test_plan["canonical_test_data"],
                "canonical_source": test_plan["canonical_source"],
                "started_unix": time.time(),
                "selection_changes_forbidden": True,
                "same_plan_resumption_allowed": True,
            },
        )

    data_verification_path = verify_local_test_fingerprints(
        args.data_root, args.output_root, test_plan, started_path
    )
    if not valid_test_data_verification(
        args.output_root, test_plan, started_path
    ):
        raise RuntimeError("Plan-bound local test-data verification failed")

    result_paths = {}
    for dataset in DATASETS:
        result_paths[dataset] = {}
        for cap in requested_caps[dataset]:
            result_paths[dataset][str(cap)] = str(
                run_final(
                    args,
                    manifest,
                    dataset,
                    cap,
                    test_plan,
                    started_path,
                )
            )

    tpe_result_paths = {}
    for dataset, objective in TPE_STUDIES:
        key = f"{dataset}_{objective}"
        tpe_result_paths[key] = str(
            run_hpo_final(
                args,
                manifest,
                tpe_locks[(dataset, objective)],
                dataset,
                objective,
                test_plan,
                started_path,
            )
        )

    selected_results = {}
    for dataset in DATASETS:
        selected_results[dataset] = {}
        for objective in ("self_scaled", "fixed_reference"):
            cap = int(
                lock["decisions"][dataset][objective]["selected_cap"]
            )
            path = Path(result_paths[dataset][str(cap)])
            result = json.loads(path.read_text(encoding="utf-8"))
            selected_results[dataset][objective] = {
                "selected_cap": cap,
                "result_path": str(path),
                "test": result["test_results"][objective],
            }
        baseline_path = Path(result_paths[dataset]["125"])
        baseline = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        selected_results[dataset]["fixed_cap_125"] = {
            "selected_cap": 125,
            "result_path": str(baseline_path),
            "test": baseline["test_results"]["fixed_reference"],
        }
        if dataset in {"FD001", "FD002"}:
            selected_results[dataset]["tpe"] = {}
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            ):
                key = f"{dataset}_{objective}"
                path = Path(tpe_result_paths[key])
                result = json.loads(path.read_text(encoding="utf-8"))
                selected_results[dataset]["tpe"][objective] = {
                    "best_params": tpe_locks[(dataset, objective)][
                        "best_params"
                    ],
                    "result_path": str(path),
                    "test": result["test_results"][
                        (
                            "self_scaled"
                            if objective == "self_scaled"
                            else "fixed_reference"
                        )
                    ],
                }

    result_hashes, test_data_fingerprints = result_integrity_payload(
        result_paths, tpe_result_paths
    )
    receipt_hashes = receipt_integrity_payload(
        result_paths,
        tpe_result_paths,
        args.output_root,
        test_plan,
        started_path,
    )
    completion = {
        "protocol": "confirmatory-final-summary-v2",
        "campaign_id": manifest["campaign_id"],
        "test_plan_sha256": stable_hash(
            json.loads(test_plan_path.read_text(encoding="utf-8"))
        ),
        "start_marker_sha256": sha256_file(started_path),
        "result_paths": result_paths,
        "tpe_result_paths": tpe_result_paths,
        "selected_results": selected_results,
        "result_file_sha256": result_hashes,
        "receipt_file_sha256": receipt_hashes,
        "test_data_verification_receipt_sha256": sha256_file(
            data_verification_path
        ),
        "test_data_fingerprints": test_data_fingerprints,
        "completed_unix": time.time(),
        "selection_source": "immutable_test_plan_only",
    }
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("campaign_id") != manifest["campaign_id"]:
            raise RuntimeError("Final summary belongs to another campaign")
    else:
        exclusive_json(summary_path, completion)
    if not completed_marker_path.exists():
        exclusive_json(
            completed_marker_path,
            completion_marker_payload(completion),
        )
    validate_completed_chain(
        completion,
        json.loads(
            completed_marker_path.read_text(encoding="utf-8")
        ),
        json.loads(test_plan_path.read_text(encoding="utf-8")),
        args.output_root,
        started_path,
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
