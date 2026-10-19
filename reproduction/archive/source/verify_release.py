"""Portable, dependency-free verification of the Zenodo release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree


OFFICIAL_MODEL_SEEDS = [45, 46, 47, 48, 49]
OFFICIAL_ARRAY_FIELDS = (
    "units",
    "raw_predictions",
    "candidate_support_predictions",
    "common_support_predictions",
    "original_targets",
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
REQUIRED_SUMMARY_CSV = {
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
EXPECTED_SUMMARY_COUNTS = {
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
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Release:
    def __init__(self, source: Path):
        self.source = source.resolve()
        self.archive = None
        if self.source.is_file():
            self.archive = zipfile.ZipFile(self.source)
            file_list = [
                name
                for name in self.archive.namelist()
                if not name.endswith("/")
            ]
            if len(file_list) != len(set(file_list)):
                raise RuntimeError("Release ZIP contains duplicate members")
            for name in file_list:
                path = Path(name.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(
                        f"Unsafe release ZIP member: {name}"
                    )
            files = set(file_list)
            roots = {name.split("/", 1)[0] for name in files}
            if len(roots) != 1:
                raise RuntimeError("Release ZIP must have one top-level root")
            self.root_name = next(iter(roots))
            prefix = self.root_name + "/"
            self.names = {
                name[len(prefix) :]
                for name in files
                if name.startswith(prefix)
            }
            self.root = None
        else:
            root = self.source
            if not (root / "MANIFEST.sha256").is_file():
                candidates = [
                    item
                    for item in root.iterdir()
                    if item.is_dir()
                    and (item / "MANIFEST.sha256").is_file()
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        "Extracted release root is ambiguous or incomplete"
                    )
                root = candidates[0]
            self.root = root
            self.root_name = root.name
            self.names = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }

    def read(self, name: str) -> bytes:
        name = name.replace("\\", "/")
        if name not in self.names:
            raise RuntimeError(f"Missing release member: {name}")
        if self.archive is not None:
            return self.archive.read(f"{self.root_name}/{name}")
        return (self.root / Path(name)).read_bytes()

    def json(self, name: str):
        return json.loads(self.read(name).decode("utf-8"))

    def close(self):
        if self.archive is not None:
            self.archive.close()


def parse_manifest(value: bytes) -> dict[str, str]:
    result = {}
    for raw_line in value.decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, name = raw_line.split("  ", 1)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or name in result
        ):
            raise RuntimeError("Malformed MANIFEST.sha256")
        result[name] = digest
    return result


def normalize_source(value: str) -> str:
    return value.replace("\\", "/").lower()


def verify_carrier_attempt_audit(release, provenance: dict, mapped) -> None:
    """Validate the optional exact-retry chain emitted by the release builder."""
    audit = provenance.get("carrier_attempt_audit")
    if audit is None:
        return
    plan_name = "confirmatory_campaign/carrier_transfer_plan.json"
    summary_name = "confirmatory_campaign/carrier_transfer_summary.json"
    plan = release.json(plan_name)
    carrier_summary = release.json(summary_name)
    campaign_id = plan.get("campaign_id")
    expected_logs = []
    for dataset in plan.get("datasets", []):
        for cap in plan.get("caps", []):
            expected_logs.append(
                "confirmatory_campaign/carrier_transfer/logs/"
                f"{dataset}_cap{int(cap)}_{campaign_id}.log"
            )
    if (
        not expected_logs
        or audit.get("original_attempt_logs") != len(expected_logs)
    ):
        raise RuntimeError("Carrier attempt-log count is invalid")
    failure_signature = (
        "FloatingPointError: Non-finite training loss"
    )
    failed_names = {
        name
        for name in expected_logs
        if failure_signature
        in release.read(name).decode("utf-8", errors="replace")
    }
    manifest_name = (
        "confirmatory_campaign/carrier_transfer_retry_manifest.json"
    )
    retry_summary_name = (
        "confirmatory_campaign/carrier_transfer_retry_summary.json"
    )
    if audit.get("retry_used") is not True:
        if (
            failed_names
            or manifest_name in release.names
            or retry_summary_name in release.names
            or audit.get("retry_attempt_logs") != 0
        ):
            raise RuntimeError("Unrecorded carrier retry/failure evidence")
        return

    manifest = release.json(manifest_name)
    retry_summary = release.json(retry_summary_name)
    terminal_failure = (
        carrier_summary.get("status") == "terminal_failure"
    )
    expected_retry_status = "failed" if terminal_failure else "complete"
    if (
        audit.get("retry_manifest_sha256")
        != sha256_bytes(release.read(manifest_name))
        or audit.get("retry_summary_sha256")
        != sha256_bytes(release.read(retry_summary_name))
        or manifest.get("protocol")
        != "carrier-transfer-exact-retry-v1"
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("carrier_plan_sha256")
        != sha256_bytes(release.read(plan_name))
        or manifest.get("retry_helper_sha256")
        != sha256_bytes(
            release.read("retry_carrier_transfer_failures.py")
        )
        or manifest.get("trainer_sha256")
        != plan.get("trainer_sha256")
        or manifest.get("model_sha256") != plan.get("model_sha256")
        or manifest.get("attempt_limit_per_cell") != 1
        or manifest.get("execution_mode") != "sequential"
        or manifest.get("parameter_changes") is not False
        or manifest.get("test_files_permitted") is not False
        or retry_summary.get("protocol") != manifest.get("protocol")
        or retry_summary.get("campaign_id") != campaign_id
        or retry_summary.get("manifest_sha256")
        != sha256_bytes(release.read(manifest_name))
        or retry_summary.get("status") != expected_retry_status
        or retry_summary.get("test_evaluated") is not False
        or audit.get("terminal_failure") is not terminal_failure
    ):
        raise RuntimeError("Bundled carrier retry chain is invalid")

    cells = manifest.get("cells", [])
    outcomes = retry_summary.get("outcomes", [])
    if (
        not cells
        or len(cells) != len(outcomes)
        or audit.get("retry_attempt_logs") != len(outcomes)
    ):
        raise RuntimeError("Bundled carrier retry cardinality is invalid")
    carrier_hashes = {
        normalize_source(path): digest
        for path, digest in carrier_summary.get(
            "result_sha256", {}
        ).items()
    }
    covered_failures = set()
    failed_retry_keys = set()
    for cell, outcome in zip(cells, outcomes, strict=True):
        try:
            identity = (cell.get("dataset"), int(cell.get("cap")))
            outcome_identity = (
                outcome.get("dataset"),
                int(outcome.get("cap")),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Bundled carrier retry identity is invalid"
            ) from exc
        if identity != outcome_identity:
            raise RuntimeError("Bundled carrier retry identity changed")
        original_source = cell.get("original_log", "")
        retry_source = outcome.get("retry_log", "")
        result_source = outcome.get("result_path", "")
        original_entry = mapped(original_source)
        retry_entry = mapped(retry_source)
        original_name = original_entry["archive_name"]
        if (
            cell.get("failure_signature") != failure_signature
            or original_name not in failed_names
            or cell.get("original_log_sha256")
            != original_entry["sha256"]
            or outcome.get("retry_log_sha256")
            != retry_entry["sha256"]
        ):
            raise RuntimeError("Bundled carrier retry evidence changed")
        if outcome.get("result_valid") is True:
            result_entry = mapped(result_source)
            if (
                outcome.get("returncode") != 0
                or outcome.get("result_sha256")
                != result_entry["sha256"]
                or carrier_hashes.get(normalize_source(result_source))
                != result_entry["sha256"]
            ):
                raise RuntimeError(
                    "Bundled successful carrier retry changed"
                )
        else:
            if (
                not terminal_failure
                or outcome.get("failure_signature_recurred") is not True
                or failure_signature
                not in release.read(
                    retry_entry["archive_name"]
                ).decode("utf-8", errors="replace")
                or "result_sha256" in outcome
            ):
                raise RuntimeError(
                    "Bundled failed carrier retry changed"
                )
            failed_retry_keys.add(identity)
        covered_failures.add(original_name)
    if covered_failures != failed_names:
        raise RuntimeError(
            "Bundled carrier retry does not cover every failed attempt"
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
            != sha256_bytes(release.read(retry_summary_name))
        ):
            raise RuntimeError(
                "Bundled terminal carrier evidence differs"
            )
    elif failed_retry_keys:
        raise RuntimeError("Bundled complete carrier summary has failures")


def per_seed_payload_valid(result: dict, canonical: dict) -> bool:
    """Reconstruct every archived scalar, summary and ensemble from arrays."""
    expected_seeds = result.get("model_seeds", [])
    expected_count = canonical.get("test_unit_count")
    canonical_target_hash = canonical.get("ordered_unit_target_sha256")
    try:
        train_cap = float(result["train_cap"])
        reporting_cap = float(result["reporting_cap"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        expected_seeds != OFFICIAL_MODEL_SEEDS
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
        or result.get("test_unit_count") != expected_count
        or not isinstance(canonical_target_hash, str)
        or len(canonical_target_hash) != 64
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

    def decision_metrics(predictions, targets):
        errors = [
            prediction - target
            for prediction, target in zip(predictions, targets, strict=True)
        ]
        values = {
            "mean_error": sum(errors) / len(errors),
            "late_prediction_fraction": (
                sum(error > 0 for error in errors) / len(errors)
            ),
            "early_prediction_fraction": (
                sum(error < 0 for error in errors) / len(errors)
            ),
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
            "own_rmse": rmse(candidate_predictions, own_targets),
            "reference_rmse": rmse(
                common_predictions, reference_targets
            ),
            "original_rmse": rmse(common_predictions, original_targets),
            "nasa_score_reference": nasa_score(
                common_predictions, reference_targets
            ),
            "nasa_score_original": nasa_score(
                common_predictions, original_targets
            ),
            "saturation_fraction": (
                sum(target > train_cap for target in original_targets)
                / len(original_targets)
            ),
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
        for metric, value in expected.items():
            candidate = observed.get(metric)
            if (
                candidate is None
                or isinstance(candidate, bool)
                or not close(candidate, value)
            ):
                return False
        return True

    test_results = result.get("test_results", {})
    for objective in ("self_scaled", "fixed_reference"):
        objective_payload = test_results.get(objective)
        if not isinstance(objective_payload, dict):
            return False
        rows = objective_payload.get("per_seed", [])
        if len(rows) != len(expected_seeds):
            return False
        scalar_values = {
            metric: [] for metric in OFFICIAL_SCALAR_METRICS
        }
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
                or any(
                    isinstance(unit, bool) or not isinstance(unit, int)
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
            if (
                not close(observed.get("mean"), expected["mean"])
                or not close(observed.get("std"), expected["std"])
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
        if (
            not isinstance(ensemble, dict)
            or set(ensemble)
            != {
                *OFFICIAL_ARRAY_FIELDS,
                *OFFICIAL_SCALAR_METRICS,
            }
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
        if not metrics_match(
            ensemble,
            reconstruct_metrics(raw_ensemble, baseline_targets),
        ):
            return False
    return (
        set(test_results) == {"self_scaled", "fixed_reference"}
        and stable_hash(
            {
                "units": baseline_units,
                "original_targets": baseline_targets,
            }
        )
        == canonical_target_hash
    )


def result_matches_plan_spec(result: dict, spec: dict, canonical: dict) -> bool:
    """Bind one release result to the immutable plan and canonical data."""
    source = result.get("source_fingerprints", {})
    data = result.get("data_fingerprints", {})
    try:
        scalar_match = (
            int(result.get("train_cap")) == int(spec.get("train_cap"))
            and math.isclose(
                float(result.get("reporting_cap")),
                float(spec.get("reporting_cap")),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and int(result.get("split_seed"))
            == int(spec.get("split_seed"))
            and int(result.get("endpoint_seed"))
            == int(spec.get("endpoint_seed"))
        )
    except (TypeError, ValueError):
        return False
    return (
        result.get("protocol") == "confirmatory-rul-cap-v2"
        and result.get("validation_only") is False
        and result.get("test_evaluated") is True
        and result.get("campaign_id") == spec.get("campaign_id")
        and result.get("dataset") == spec.get("dataset")
        and result.get("tag") == spec.get("tag")
        and scalar_match
        and result.get("model_seeds") == spec.get("model_seeds")
        and result.get("config") == spec.get("config")
        and source.get("trainer_sha256") == spec.get("trainer_sha256")
        and source.get("model_sha256") == spec.get("model_sha256")
        and data.get("train_sha256") == spec.get("train_sha256")
        and data.get("test_sha256") == canonical.get("test_sha256")
        and data.get("truth_sha256") == canonical.get("truth_sha256")
        and per_seed_payload_valid(result, canonical)
    )


def verify(source: Path) -> dict:
    release = Release(source)
    try:
        manifest = parse_manifest(release.read("MANIFEST.sha256"))
        expected_names = set(manifest) | {"MANIFEST.sha256"}
        if release.names != expected_names:
            missing = sorted(expected_names - release.names)
            extra = sorted(release.names - expected_names)
            raise RuntimeError(
                f"Release member set differs from manifest; "
                f"missing={missing}, extra={extra}"
            )
        for name, expected in manifest.items():
            if sha256_bytes(release.read(name)) != expected:
                raise RuntimeError(f"Manifest hash mismatch: {name}")

        provenance = release.json("RELEASE_PROVENANCE.json")
        if provenance.get("protocol") != "ress-release-provenance-v1":
            raise RuntimeError("Release provenance protocol is invalid")
        before = provenance.get(
            "archive_file_sha256_before_provenance", {}
        )
        expected_before_names = release.names - {
            "MANIFEST.sha256",
            "RELEASE_PROVENANCE.json",
        }
        if set(before) != expected_before_names:
            raise RuntimeError(
                "Release provenance does not cover the exact pre-provenance "
                "member set"
            )
        source_map = provenance.get("source_to_archive", {})
        archive_names = [
            entry.get("archive_name") for entry in source_map.values()
        ]
        if (
            not source_map
            or any(not name for name in archive_names)
            or len(archive_names) != len(set(archive_names))
        ):
            raise RuntimeError(
                "Release source mapping is empty or not one-to-one"
            )
        source_root = normalize_source(
            provenance.get("source_campaign_root", "")
        ).rstrip("/")
        if not source_root:
            raise RuntimeError("Release provenance lacks campaign root")
        for source_name, entry in source_map.items():
            normalized = normalize_source(source_name)
            if normalized.startswith(source_root + "/"):
                suffix = normalized[len(source_root) + 1 :]
                expected_archive = (
                    "confirmatory_campaign/" + suffix
                )
                if normalize_source(entry["archive_name"]) != expected_archive:
                    raise RuntimeError(
                        "Campaign evidence mapping is not root-relative: "
                        f"{source_name}"
                    )
        aliases = {
            normalize_source(source_name): entry
            for source_name, entry in source_map.items()
        }

        def mapped(source_name: str) -> dict:
            entry = source_map.get(source_name)
            if entry is None:
                entry = aliases.get(normalize_source(source_name))
            if entry is None:
                raise RuntimeError(
                    f"Absolute source path has no portable mapping: {source_name}"
                )
            observed = sha256_bytes(release.read(entry["archive_name"]))
            if observed != entry["sha256"]:
                raise RuntimeError(
                    f"Mapped evidence hash mismatch: {entry['archive_name']}"
                )
            return entry

        for name, expected in before.items():
            if sha256_bytes(release.read(name)) != expected:
                raise RuntimeError(
                    f"Pre-provenance release hash mismatch: {name}"
                )
        for source_name in source_map:
            mapped(source_name)
        verify_carrier_attempt_audit(release, provenance, mapped)

        premarker_entries = provenance.get("premarker_provenance", {})
        expected_premarker_names = {
            "selection_attempt_manifest": (
                "confirmatory_campaign/SELECTION_ATTEMPT_MANIFEST.json"
            ),
            "qa_manifest": (
                "confirmatory_campaign/qa/pytest_run_manifest.json"
            ),
            "qa_junit": "confirmatory_campaign/qa/pytest-junit.xml",
            "premarker_campaign_inventory": (
                "confirmatory_campaign/PREMARKER_CAMPAIGN_INVENTORY.json"
            ),
        }
        if set(premarker_entries) != set(expected_premarker_names):
            raise RuntimeError(
                "Release provenance lacks the exact pre-marker evidence set"
            )
        for key, expected_name in expected_premarker_names.items():
            entry = premarker_entries[key]
            if entry.get("archive_name") != expected_name:
                raise RuntimeError(
                    f"Pre-marker evidence mapping differs: {key}"
                )
            if (
                expected_name not in release.names
                or sha256_bytes(release.read(expected_name))
                != entry.get("sha256")
            ):
                raise RuntimeError(
                    f"Pre-marker evidence hash differs: {key}"
                )
        selection_attempt = release.json(
            expected_premarker_names["selection_attempt_manifest"]
        )
        qa_manifest = release.json(
            expected_premarker_names["qa_manifest"]
        )
        qa_junit_bytes = release.read(
            expected_premarker_names["qa_junit"]
        )
        premarker_inventory = release.json(
            expected_premarker_names["premarker_campaign_inventory"]
        )
        selection_core = dict(selection_attempt)
        selection_self = selection_core.pop(
            "manifest_sha256_without_self", None
        )
        inventory_core = dict(premarker_inventory)
        inventory_self = inventory_core.pop(
            "inventory_sha256_without_self", None
        )
        if (
            selection_attempt.get("protocol")
            != "selection-attempt-provenance-v1"
            or selection_self != stable_hash(selection_core)
            or premarker_inventory.get("protocol")
            != "premarker-campaign-inventory-v1"
            or premarker_inventory.get("test_evidence_detected") is not False
            or inventory_self != stable_hash(inventory_core)
            or qa_manifest.get("protocol") != "artifact-qa-receipt-v1"
            or qa_manifest.get("exit_code") != 0
            or qa_manifest.get("junit_sha256")
            != sha256_bytes(qa_junit_bytes)
        ):
            raise RuntimeError("Bundled pre-marker evidence is invalid")
        junit_root = ElementTree.fromstring(qa_junit_bytes)
        junit_suites = (
            [junit_root]
            if junit_root.tag.rsplit("}", 1)[-1] == "testsuite"
            else [
                child
                for child in junit_root
                if child.tag.rsplit("}", 1)[-1] == "testsuite"
            ]
        )
        qa_counts = {
            key: sum(
                int(suite.attrib.get(key, 0))
                for suite in junit_suites
            )
            for key in ("tests", "failures", "errors", "skipped")
        }
        qa_counts["passed"] = (
            qa_counts["tests"]
            - qa_counts["failures"]
            - qa_counts["errors"]
            - qa_counts["skipped"]
        )
        if (
            qa_manifest.get("counts") != qa_counts
            or qa_counts["failures"] != 0
            or qa_counts["errors"] != 0
            or qa_counts["passed"] < 1
            or qa_manifest.get("environment_lock_sha256")
            != sha256_bytes(release.read("environment-lock.txt"))
            or qa_manifest.get("qa_runner_sha256")
            != sha256_bytes(release.read("run_qa_suite.py"))
        ):
            raise RuntimeError("Bundled QA receipt is invalid")
        qa_sources = qa_manifest.get("source_file_sha256", {})
        bundled_python = {
            name
            for name in release.names
            if name.endswith(".py")
            and not name.startswith("confirmatory_campaign/")
        }
        if any(
            qa_sources.get(name) != sha256_bytes(release.read(name))
            for name in bundled_python
        ):
            raise RuntimeError(
                "Bundled Python source differs from the tested source tree"
            )
        selection_sources = selection_attempt.get("source_sha256", {})
        if (
            selection_sources.get("manifest_builder")
            != sha256_bytes(
                release.read("create_selection_attempt_manifest.py")
            )
            or selection_sources.get("canonical_cap_runner")
            != sha256_bytes(
                release.read("run_confirmatory_cap_sweep.py")
            )
            or selection_sources.get("frozen_trainer")
            != sha256_bytes(
                release.read("confirmatory_rul_train.py")
            )
        ):
            raise RuntimeError(
                "Bundled selection-attempt sources differ from provenance"
            )
        for relative, record in premarker_inventory.get(
            "files", {}
        ).items():
            archive_name = "confirmatory_campaign/" + relative
            if (
                archive_name not in release.names
                or len(release.read(archive_name))
                != record.get("bytes")
                or sha256_bytes(release.read(archive_name))
                != record.get("sha256")
            ):
                raise RuntimeError(
                    "Bundled pre-marker inventory file differs: "
                    + relative
                )

        preflight_entry = provenance["preflight_report"]
        preflight = release.json(preflight_entry["archive_name"])
        preflight_core = dict(preflight)
        preflight_hash = preflight_core.pop(
            "report_sha256_without_self", None
        )
        if (
            preflight.get("status") != "pass"
            or preflight.get("test_or_truth_files_opened") is not False
            or stable_hash(preflight_core) != preflight_hash
        ):
            raise RuntimeError("Bundled preflight report is invalid")
        for source_name, expected in preflight.get(
            "artefact_file_sha256", {}
        ).items():
            entry = mapped(source_name)
            if entry["sha256"] != expected:
                raise RuntimeError(
                    f"Preflight evidence differs in release: {source_name}"
                )

        plan = release.json(provenance["test_plan"]["archive_name"])
        started = release.json(
            provenance["test_start_marker"]["archive_name"]
        )
        data_verified = release.json(
            provenance["test_data_verification"]["archive_name"]
        )
        summary = release.json(provenance["final_summary"]["archive_name"])
        marker = release.json(
            provenance["test_completion_marker"]["archive_name"]
        )
        if (
            summary.get("protocol") != "confirmatory-final-summary-v2"
            or summary.get("test_plan_sha256") != stable_hash(plan)
            or summary.get("start_marker_sha256")
            != provenance["test_start_marker"]["sha256"]
            or summary.get("test_data_verification_receipt_sha256")
            != provenance["test_data_verification"]["sha256"]
            or marker.get("campaign_id") != summary.get("campaign_id")
            or marker.get("final_summary_sha256") != stable_hash(summary)
        ):
            raise RuntimeError("Bundled official-evaluation chain is invalid")
        expected_verified_files = {
            dataset: {
                "test_filename": value["test_filename"],
                "test_sha256": value["test_sha256"],
                "truth_filename": value["truth_filename"],
                "truth_sha256": value["truth_sha256"],
            }
            for dataset, value in plan["canonical_test_data"].items()
        }
        if (
            data_verified.get("protocol")
            != "confirmatory-local-test-data-verification-v2"
            or data_verified.get("campaign_id") != plan.get("campaign_id")
            or data_verified.get("test_plan_sha256") != stable_hash(plan)
            or data_verified.get("start_marker_sha256")
            != provenance["test_start_marker"]["sha256"]
            or data_verified.get("canonical_manifest_sha256")
            != plan["canonical_source"]["canonical_manifest_sha256"]
            or data_verified.get("verified_files") != expected_verified_files
        ):
            raise RuntimeError(
                "Bundled local test-data verification is invalid"
            )
        if (
            plan.get("protocol") != "confirmatory-test-plan-v2"
            or started.get("campaign_id") != plan.get("campaign_id")
            or started.get("test_plan_sha256") != stable_hash(plan)
            or started.get("selection_changes_forbidden") is not True
            or summary.get("campaign_id") != plan.get("campaign_id")
            or provenance.get("campaign_id") != plan.get("campaign_id")
        ):
            raise RuntimeError(
                "Bundled plan/start/campaign identities are inconsistent"
            )
        expected_premarker_hashes = {
            "selection_attempt_manifest_sha256": stable_hash(
                selection_attempt
            ),
            "qa_manifest_sha256": stable_hash(qa_manifest),
            "qa_junit_sha256": sha256_bytes(qa_junit_bytes),
            "premarker_campaign_inventory_sha256": stable_hash(
                premarker_inventory
            ),
        }
        if any(
            plan.get(key) != value or preflight.get(key) != value
            for key, value in expected_premarker_hashes.items()
        ):
            raise RuntimeError(
                "Immutable plan/preflight pre-marker bindings differ"
            )

        campaign_manifest = release.json(
            "confirmatory_campaign/campaign_manifest.json"
        )
        bundled_hashes = {
            "preflight_report_sha256": stable_hash(preflight),
            "final_runner_sha256": sha256_bytes(
                release.read("run_confirmatory_final.py")
            ),
            "final_attested_trainer_sha256": sha256_bytes(
                release.read("final_attested_trainer.py")
            ),
            "summarizer_sha256": sha256_bytes(
                release.read("summarize_confirmatory.py")
            ),
            "manuscript_asset_generator_sha256": sha256_bytes(
                release.read("generate_manuscript_assets.py")
            ),
            "trainer_sha256": sha256_bytes(
                release.read("confirmatory_rul_train.py")
            ),
            "preflight_script_sha256": sha256_bytes(
                release.read("preflight_confirmatory.py")
            ),
            "canonical_source_manifest_sha256": sha256_bytes(
                release.read("canonical_cmapss_sha256.json")
            ),
            "model_sha256": sha256_bytes(
                release.read("baselines/models_v3.py")
            ),
        }
        expected_hashes = {
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
            name: {
                "expected": expected_hashes[name],
                "observed": value,
            }
            for name, value in bundled_hashes.items()
            if expected_hashes[name] != value
        }
        if (
            campaign_manifest.get("protocol")
            != "confirmatory-cap-sweep-v2"
            or campaign_manifest.get("campaign_id")
            != plan.get("campaign_id")
            or preflight.get("screening_campaign_id")
            != plan.get("campaign_id")
            or campaign_manifest.get("trainer_sha256")
            != plan.get("trainer_sha256")
        ):
            mismatches["campaign_identity"] = {
                "manifest_campaign_id": campaign_manifest.get(
                    "campaign_id"
                ),
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
                "Bundled source differs from immutable test-plan bindings: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if provenance.get("source_bindings") != bundled_hashes:
            raise RuntimeError(
                "Release provenance source bindings are incomplete or stale"
            )

        planned_specs = {}
        for dataset_specs in plan.get("cap_result_specs", {}).values():
            for spec in dataset_specs.values():
                planned_specs[spec["result_filename"]] = spec
        for spec in plan.get("tpe_result_specs", {}).values():
            if spec["result_filename"] in planned_specs:
                raise RuntimeError(
                    "Immutable plan contains duplicate result filenames"
                )
            planned_specs[spec["result_filename"]] = spec
        result_hashes = summary.get("result_file_sha256", {})
        result_entries = {
            mapped(source_name)["archive_name"]: (
                source_name,
                expected,
            )
            for source_name, expected in result_hashes.items()
        }
        expected_result_members = {
            "confirmatory_campaign/results/" + filename
            for filename in planned_specs
        }
        if set(result_entries) != expected_result_members:
            raise RuntimeError(
                "Bundled result set differs from immutable test plan"
            )
        receipt_hashes = summary.get("receipt_file_sha256", {})
        receipt_entries = {
            mapped(source_name)["archive_name"]: (
                source_name,
                expected,
            )
            for source_name, expected in receipt_hashes.items()
        }
        expected_receipt_members = {
            "confirmatory_campaign/final_receipts/"
            + filename
            + ".receipt.json"
            for filename in planned_specs
        }
        if set(receipt_entries) != expected_receipt_members:
            raise RuntimeError(
                "Bundled receipt set differs from immutable test plan"
            )

        for source_name, expected in result_hashes.items():
            entry = mapped(source_name)
            if entry["sha256"] != expected:
                raise RuntimeError(
                    f"Final result hash differs in release: {source_name}"
                )
            result = release.json(entry["archive_name"])
            test_results = result.get("test_results", {})
            if not test_results:
                raise RuntimeError(
                    f"Final result lacks test outputs: {entry['archive_name']}"
                )
            filename = entry["archive_name"].rsplit("/", 1)[-1]
            spec = planned_specs[filename]
            try:
                canonical = plan["canonical_test_data"][spec["dataset"]]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Immutable plan lacks canonical test binding: {filename}"
                ) from exc
            if not result_matches_plan_spec(result, spec, canonical):
                raise RuntimeError(
                    "Final result fails plan-bound semantic reconstruction: "
                    f"validation: {entry['archive_name']}"
                )
            receipt_name = (
                "confirmatory_campaign/final_receipts/"
                + filename
                + ".receipt.json"
            )
            launch_name = receipt_name.replace(
                ".receipt.json", ".launch.json"
            )
            receipt = release.json(receipt_name)
            try:
                receipt_completed_unix = float(
                    receipt.get("completed_unix")
                )
            except (TypeError, ValueError):
                receipt_completed_unix = math.nan
            launch = release.json(launch_name)
            launch_hash = sha256_bytes(release.read(launch_name))
            start_hash = provenance["test_start_marker"]["sha256"]
            nonce = launch.get("launch_nonce", "")
            if (
                launch.get("protocol") != "confirmatory-final-launch-v2"
                or normalize_source(launch.get("result_path", ""))
                != normalize_source(source_name)
                or launch.get("test_plan_sha256") != stable_hash(plan)
                or launch.get("start_marker_sha256") != start_hash
                or launch.get("expected_result") != spec
                or launch.get("result_absent_before_authorization")
                is not True
                or len(nonce) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in nonce
                )
            ):
                raise RuntimeError(
                    f"Final launch authorization is invalid: {launch_name}"
                )
            attestation = result.get(
                "official_execution_attestation", {}
            )
            expected_attestation = {
                "protocol": (
                    "confirmatory-final-execution-attestation-v2"
                ),
                "test_plan_sha256": stable_hash(plan),
                "start_marker_sha256": start_hash,
                "launch_authorization_sha256": launch_hash,
                "launch_nonce": nonce,
            }
            if attestation != expected_attestation:
                raise RuntimeError(
                    f"Final execution attestation is invalid: "
                    f"{entry['archive_name']}"
                )
            receipt_entry = receipt_entries[receipt_name]
            if sha256_bytes(release.read(receipt_name)) != receipt_entry[1]:
                raise RuntimeError(
                    f"Final receipt hash differs in release: {receipt_name}"
                )
            receipt_core = dict(receipt)
            receipt_core.pop("completed_unix", None)
            attempt_hashes = receipt.get("attempt_file_sha256", {})
            if not isinstance(attempt_hashes, dict) or not attempt_hashes:
                raise RuntimeError(
                    f"Final attempt ledger is invalid: {receipt_name}"
                )
            attempt_records = {}
            for attempt_source, attempt_hash in attempt_hashes.items():
                attempt_entry = mapped(attempt_source)
                if (
                    attempt_entry["sha256"] != attempt_hash
                    or not attempt_entry["archive_name"].startswith(
                        "confirmatory_campaign/final_attempts/"
                    )
                ):
                    raise RuntimeError(
                        f"Final attempt ledger differs in release: "
                        f"{attempt_source}"
                    )
                attempt_name = attempt_entry["archive_name"].rsplit(
                    "/", 1
                )[-1]
                if attempt_name in attempt_records:
                    raise RuntimeError(
                        f"Duplicate final-attempt record: {attempt_name}"
                    )
                attempt_records[attempt_name] = (
                    attempt_entry,
                    release.json(attempt_entry["archive_name"]),
                )
            starts = {}
            completes = {}
            expected_prefix = filename + ".attempt"
            for attempt_name, (attempt_entry, payload) in (
                attempt_records.items()
            ):
                if not attempt_name.startswith(expected_prefix):
                    raise RuntimeError(
                        f"Final-attempt record belongs to another result: "
                        f"{attempt_name}"
                    )
                protocol = payload.get("protocol")
                if protocol == "confirmatory-final-attempt-start-v2":
                    attempt = payload.get("attempt")
                    expected_name = (
                        f"{expected_prefix}{attempt:03d}.start.json"
                        if isinstance(attempt, int)
                        and not isinstance(attempt, bool)
                        and attempt > 0
                        else ""
                    )
                    command = payload.get("command")
                    try:
                        started_unix = float(payload.get("started_unix"))
                    except (TypeError, ValueError):
                        started_unix = math.nan
                    if (
                        attempt_name != expected_name
                        or attempt in starts
                        or set(payload)
                        != {
                            "protocol",
                            "attempt",
                            "result_path",
                            "test_plan_sha256",
                            "start_marker_sha256",
                            "launch_authorization_sha256",
                            "command",
                            "command_sha256",
                            "log_path",
                            "started_unix",
                        }
                        or normalize_source(payload.get("result_path", ""))
                        != normalize_source(source_name)
                        or payload.get("test_plan_sha256")
                        != stable_hash(plan)
                        or payload.get("start_marker_sha256") != start_hash
                        or payload.get("launch_authorization_sha256")
                        != launch_hash
                        or not isinstance(command, list)
                        or not command
                        or any(not isinstance(item, str) for item in command)
                        or payload.get("command_sha256")
                        != stable_hash({"command": command})
                        or not math.isfinite(started_unix)
                    ):
                        raise RuntimeError(
                            f"Final-attempt start is invalid: {attempt_name}"
                        )
                    log_entry = mapped(payload.get("log_path", ""))
                    if not log_entry["archive_name"].startswith(
                        "confirmatory_campaign/final_logs/"
                    ):
                        raise RuntimeError(
                            f"Final-attempt log is outside final_logs: "
                            f"{attempt_name}"
                        )
                    starts[attempt] = (
                        attempt_entry,
                        payload,
                        log_entry,
                    )
                elif protocol == "confirmatory-final-attempt-complete-v2":
                    start_name = attempt_name.replace(
                        ".complete.json", ".start.json"
                    )
                    if (
                        not attempt_name.endswith(".complete.json")
                        or start_name == attempt_name
                        or start_name in completes
                        or set(payload)
                        != {
                            "protocol",
                            "attempt_start_sha256",
                            "returncode",
                            "log_sha256",
                            "completed_unix",
                        }
                    ):
                        raise RuntimeError(
                            f"Final-attempt completion is invalid: "
                            f"{attempt_name}"
                        )
                    completes[start_name] = payload
                else:
                    raise RuntimeError(
                        f"Unknown final-attempt protocol: {attempt_name}"
                    )
            if sorted(starts) != list(range(1, len(starts) + 1)):
                raise RuntimeError(
                    f"Final-attempt starts are not contiguous: {filename}"
                )
            start_names = {
                f"{expected_prefix}{attempt:03d}.start.json": values
                for attempt, values in starts.items()
            }
            if not set(completes).issubset(start_names):
                raise RuntimeError(
                    f"Final-attempt completion lacks a start: {filename}"
                )
            successful_completions = []
            for start_name, completion in completes.items():
                start_entry, start_payload, log_entry = start_names[
                    start_name
                ]
                try:
                    completed_unix = float(completion.get("completed_unix"))
                except (TypeError, ValueError):
                    completed_unix = math.nan
                returncode = completion.get("returncode")
                if (
                    completion.get("attempt_start_sha256")
                    != start_entry["sha256"]
                    or isinstance(returncode, bool)
                    or not isinstance(returncode, int)
                    or completion.get("log_sha256") != log_entry["sha256"]
                    or not math.isfinite(completed_unix)
                    or completed_unix
                    < float(start_payload["started_unix"])
                ):
                    raise RuntimeError(
                        f"Final-attempt completion cannot be verified: "
                        f"{start_name}"
                    )
                if returncode == 0:
                    successful_completions.append(completed_unix)
            if (
                not successful_completions
                or not math.isfinite(receipt_completed_unix)
                or receipt_completed_unix < max(successful_completions)
            ):
                raise RuntimeError(
                    f"Final receipt lacks a chronologically prior successful "
                    f"attempt: {receipt_name}"
                )
            expected_receipt = {
                "protocol": "confirmatory-final-result-receipt-v2",
                "result_path": source_name,
                "result_sha256": expected,
                "test_plan_sha256": stable_hash(plan),
                "start_marker_sha256": start_hash,
                "launch_authorization_sha256": launch_hash,
                "expected_result": spec,
                "test_data_fingerprints": {
                    "test_sha256": result["data_fingerprints"][
                        "test_sha256"
                    ],
                    "truth_sha256": result["data_fingerprints"][
                        "truth_sha256"
                    ],
                },
                "attempt_file_sha256": attempt_hashes,
            }
            if receipt_core != expected_receipt:
                raise RuntimeError(
                    f"Final result receipt is invalid: {receipt_name}"
                )

        status_name = (
            "confirmatory_campaign/summaries/summary_status.json"
        )
        status = release.json(status_name)
        if (
            status.get("protocol") != "confirmatory-summary-v2"
            or status.get("complete") is not True
            or status.get("bound_failures") != 0
            or any(
                status.get(name) != value
                for name, value in EXPECTED_SUMMARY_COUNTS.items()
            )
            or status.get("final_test_summary_sha256")
            != sha256_bytes(release.read(
                provenance["final_summary"]["archive_name"]
            ))
            or status.get("test_evaluated_marker_sha256")
            != sha256_bytes(release.read(
                provenance["test_completion_marker"]["archive_name"]
            ))
            or status.get("summarizer_sha256")
            != sha256_bytes(release.read("summarize_confirmatory.py"))
            or status.get("manuscript_asset_generator_sha256")
            != sha256_bytes(release.read("generate_manuscript_assets.py"))
        ):
            raise RuntimeError("Bundled manuscript summary is incomplete")
        csv_hashes = status.get("csv_sha256", {})
        if set(csv_hashes) != REQUIRED_SUMMARY_CSV:
            raise RuntimeError(
                "Bundled summary CSV set is incomplete or unplanned"
            )
        for name, expected in csv_hashes.items():
            archive_name = f"confirmatory_campaign/summaries/{name}"
            if sha256_bytes(release.read(archive_name)) != expected:
                raise RuntimeError(
                    f"Bundled summary CSV hash mismatch: {archive_name}"
                )

        return {
            "status": "pass",
            "protocol": provenance["protocol"],
            "artifact_version": provenance["artifact_version"],
            "campaign_id": provenance["campaign_id"],
            "manifest_files": len(manifest),
            "portable_source_mappings": len(source_map),
        }
    finally:
        release.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "release",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Release ZIP or extracted release directory",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.release), indent=2))


if __name__ == "__main__":
    main()
