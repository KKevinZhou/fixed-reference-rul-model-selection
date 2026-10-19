"""Generate auditable CSV/LaTeX summaries from immutable campaign artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from preflight_confirmatory import run_preflight
from run_confirmatory_final import (
    config_matches,
    stable_hash,
    validate_completed_chain,
)


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
CAPS = [90, 100, 110, 115, 125]
SPLITS = [20260724, 20260725, 20260726]
FINAL_MODEL_SEEDS = {45, 46, 47, 48, 49}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rmse(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mse(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((prediction - target) ** 2))


def validated_prediction_arrays(metrics, cap, context):
    """Load archived arrays and prove both reported supports derive from one raw output."""
    units = tuple(str(unit) for unit in metrics["units"])
    raw_prediction = np.asarray(
        metrics["raw_predictions"], dtype=np.float64
    )
    candidate_prediction = np.asarray(
        metrics["candidate_support_predictions"], dtype=np.float64
    )
    common_prediction = np.asarray(
        metrics["common_support_predictions"], dtype=np.float64
    )
    original_target = np.asarray(
        metrics["original_targets"], dtype=np.float64
    )
    arrays = (
        raw_prediction,
        candidate_prediction,
        common_prediction,
        original_target,
    )
    if (
        len(units) != len(set(units))
        or any(len(array) != len(units) for array in arrays)
        or not np.all(np.isfinite(np.concatenate(arrays)))
    ):
        raise RuntimeError(f"Invalid prediction arrays in {context}")
    expected_candidate = np.clip(raw_prediction, 0.0, float(cap))
    expected_common = np.clip(raw_prediction, 0.0, 125.0)
    if not np.allclose(
        candidate_prediction,
        expected_candidate,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Candidate-support predictions are not a clip of raw outputs in {context}"
        )
    if not np.allclose(
        common_prediction,
        expected_common,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Common-support predictions are not a clip of raw outputs in {context}"
        )
    return units, candidate_prediction, common_prediction, original_target


def validate_unit_target_anchor(anchors, key, units, targets, context):
    """Require identical held-out units and targets across caps and objectives."""
    if key not in anchors:
        anchors[key] = (units, targets.copy())
        return
    expected_units, expected_targets = anchors[key]
    if (
        units != expected_units
        or targets.shape != expected_targets.shape
        or not np.array_equal(targets, expected_targets)
    ):
        raise RuntimeError(
            f"Held-out unit ordering or original targets changed in {context}"
        )


def nasa_score(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    terms = np.where(
        error < 0,
        np.exp(np.minimum(-error / 13.0, 50.0)) - 1.0,
        np.exp(np.minimum(error / 10.0, 50.0)) - 1.0,
    )
    return float(np.sum(terms))


def reference_decision_metrics(prediction, reference_target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(reference_target, dtype=np.float64)
    error = prediction - target
    output = {
        "mean_error": float(np.mean(error)),
        "late_prediction_fraction": float(np.mean(error > 0)),
    }
    for threshold in (20, 30, 50):
        due = target <= threshold
        predicted_due = prediction <= threshold
        due_count = int(np.sum(due))
        not_due_count = int(np.sum(~due))
        if due_count == 0 or not_due_count == 0:
            raise ValueError(
                f"Maintenance threshold {threshold} has an empty target stratum"
            )
        output[f"n_due_at_{threshold}"] = due_count
        output[f"n_not_due_at_{threshold}"] = not_due_count
        output[f"delayed_action_rate_at_{threshold}"] = float(
            np.sum(due & ~predicted_due) / due_count
        )
        output[f"premature_action_rate_at_{threshold}"] = float(
            np.sum(~due & predicted_due) / not_due_count
        )
    return output


def screening_rows(root: Path, allow_incomplete=False):
    rows = []
    bound_rows = []
    missing = []
    unit_target_anchors = {}
    manifest_path = root / "campaign_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    for dataset in DATASETS:
        for split in SPLITS:
            for cap in CAPS:
                path = (
                    root
                    / "results"
                    / f"{dataset}_cap{cap}_split{split}.json"
                )
                if not path.exists():
                    missing.append(str(path))
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if (
                    manifest is None
                    or result.get("campaign_id")
                    != manifest.get("campaign_id")
                    or result.get("dataset") != dataset
                    or int(result.get("train_cap")) != cap
                    or int(result.get("split_seed")) != split
                    or result.get("model_seeds") != [42]
                    or result.get("validation_only") is not True
                    or result.get("test_evaluated") is not False
                    or result.get("source_fingerprints", {}).get(
                        "trainer_sha256"
                    )
                    != manifest.get("trainer_sha256")
                ):
                    raise RuntimeError(
                        f"Screening result failed campaign validation: {path}"
                    )
                for seed_result in result["per_seed_validation"]:
                    seed = seed_result["seed"]
                    for objective, objective_metric in (
                        ("self_scaled", "own_rmse"),
                        ("fixed_reference", "reference_rmse"),
                    ):
                        metrics = seed_result["checkpoints"][objective][
                            "metrics"
                        ]
                        (
                            units,
                            candidate_prediction,
                            common_prediction,
                            original_target,
                        ) = validated_prediction_arrays(metrics, cap, path)
                        validate_unit_target_anchor(
                            unit_target_anchors,
                            (dataset, split, int(seed)),
                            units,
                            original_target,
                            path,
                        )
                        candidate_target = np.clip(
                            original_target, 0.0, float(cap)
                        )
                        reference_target = np.clip(
                            original_target, 0.0, 125.0
                        )
                        recomputed_own = rmse(
                            candidate_prediction,
                            candidate_target,
                        )
                        tolerance = 1e-8 * (
                            1.0 + abs(recomputed_own)
                        )
                        if (
                            abs(
                                recomputed_own
                                - float(metrics["own_rmse"])
                            )
                            > tolerance
                        ):
                            raise RuntimeError(
                                f"Stored own RMSE is inconsistent in {path}"
                            )
                        candidate_original_rmse = rmse(
                            candidate_prediction,
                            original_target,
                        )
                        recomputed_saturation = float(
                            np.mean(original_target > cap)
                        )
                        recomputed_exposure = float(
                            np.sqrt(
                                np.mean(
                                    np.maximum(
                                        original_target - cap, 0.0
                                    )
                                    ** 2
                                )
                            )
                        )
                        recomputed_reference_exposure = float(
                            np.sqrt(
                                np.mean(
                                    np.maximum(
                                        np.minimum(original_target, 125.0)
                                        - cap,
                                        0.0,
                                    )
                                    ** 2
                                )
                            )
                        )
                        if (
                            abs(
                                recomputed_saturation
                                - float(metrics["saturation_fraction"])
                            )
                            > 1e-12
                            or abs(
                                recomputed_exposure
                                - float(metrics["exposure_bound"])
                            )
                            > 1e-8
                            * (1.0 + abs(recomputed_exposure))
                        ):
                            raise RuntimeError(
                                "Stored saturation/exposure is "
                                f"inconsistent in {path}"
                            )
                        candidate_reference_rmse = rmse(
                            candidate_prediction,
                            reference_target,
                        )
                        common_candidate_target_rmse = rmse(
                            common_prediction,
                            candidate_target,
                        )
                        common_reference_rmse = rmse(
                            common_prediction,
                            reference_target,
                        )
                        if abs(
                            common_reference_rmse
                            - float(metrics["reference_rmse"])
                        ) > 1e-8 * (1.0 + abs(common_reference_rmse)):
                            raise RuntimeError(
                                "Stored common-support reference RMSE is "
                                f"inconsistent in {path}"
                            )
                        common_original_rmse = rmse(
                            common_prediction,
                            original_target,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "split_seed": split,
                                "model_seed": seed,
                                "cap": cap,
                                "objective": objective,
                                "selection_metric": objective_metric,
                                "selection_value": (
                                    recomputed_own
                                    if objective_metric == "own_rmse"
                                    else common_reference_rmse
                                ),
                                "own_rmse": recomputed_own,
                                "candidate_target_common_support_rmse": (
                                    common_candidate_target_rmse
                                ),
                                "reference_rmse": common_reference_rmse,
                                "candidate_support_reference_rmse": (
                                    candidate_reference_rmse
                                ),
                                "candidate_target_candidate_support_mse": mse(
                                    candidate_prediction, candidate_target
                                ),
                                "candidate_target_common_support_mse": mse(
                                    common_prediction, candidate_target
                                ),
                                "reference_target_candidate_support_mse": mse(
                                    candidate_prediction, reference_target
                                ),
                                "reference_target_common_support_mse": mse(
                                    common_prediction, reference_target
                                ),
                                "candidate_original_rmse": candidate_original_rmse,
                                "common_original_rmse": common_original_rmse,
                                "saturation_fraction": (
                                    recomputed_saturation
                                ),
                                "exposure_bound": recomputed_exposure,
                                "reference_exposure_bound": (
                                    recomputed_reference_exposure
                                ),
                                "best_epoch": seed_result["checkpoints"][
                                    objective
                                ]["best_epoch"],
                                "result_path": str(path),
                            }
                        )
                        hidden_gap = (
                            candidate_original_rmse**2
                            - recomputed_own**2
                        )
                        bound_squared = recomputed_exposure**2
                        bound_rows.append(
                            {
                                "dataset": dataset,
                                "split_seed": split,
                                "model_seed": seed,
                                "cap": cap,
                                "objective": objective,
                                "candidate_hidden_mse_gap": hidden_gap,
                                "exposure_bound_squared": bound_squared,
                                "bound_satisfied": hidden_gap
                                >= bound_squared
                                - 1e-8
                                * (
                                    1.0
                                    + abs(hidden_gap)
                                    + abs(bound_squared)
                                ),
                                "bound_fraction_of_hidden_gap": (
                                    bound_squared / hidden_gap
                                    if hidden_gap > 0
                                    else None
                                ),
                                "bound_applicable_to": "candidate_support",
                            }
                        )
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Incomplete screening matrix: missing {len(missing)} files; "
            f"first is {missing[0]}"
        )
    return rows, bound_rows


def stability_rows(root: Path, allow_incomplete=False):
    """Recompute the additional primary-split seeds from archived arrays."""
    rows = []
    missing = []
    unit_target_anchors = {}
    manifest_path = root / "stability_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    for dataset in DATASETS:
        for cap in CAPS:
            path = (
                root
                / "results"
                / (
                    f"{dataset}_stability_cap{cap}_split20260724_"
                    "seeds43-44.json"
                )
            )
            if not path.exists():
                missing.append(str(path))
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest is None
                or result.get("campaign_id") != manifest.get("campaign_id")
                or result.get("dataset") != dataset
                or int(result.get("train_cap")) != cap
                or int(result.get("split_seed")) != 20260724
                or result.get("model_seeds") != [43, 44]
                or result.get("validation_only") is not True
                or result.get("test_evaluated") is not False
                or result.get("source_fingerprints", {}).get(
                    "trainer_sha256"
                )
                != manifest.get("trainer_sha256")
            ):
                raise RuntimeError(
                    f"Stability result failed campaign validation: {path}"
                )
            for seed_result in result["per_seed_validation"]:
                seed = int(seed_result["seed"])
                for objective, selection_metric in (
                    ("self_scaled", "own_rmse"),
                    ("fixed_reference", "reference_rmse"),
                ):
                    metrics = seed_result["checkpoints"][objective]["metrics"]
                    (
                        units,
                        candidate_prediction,
                        common_prediction,
                        original_target,
                    ) = validated_prediction_arrays(metrics, cap, path)
                    validate_unit_target_anchor(
                        unit_target_anchors,
                        (dataset, 20260724, seed),
                        units,
                        original_target,
                        path,
                    )
                    candidate_target = np.clip(
                        original_target, 0.0, float(cap)
                    )
                    reference_target = np.clip(
                        original_target, 0.0, 125.0
                    )
                    own_value = rmse(
                        candidate_prediction,
                        candidate_target,
                    )
                    candidate_target_common_support = rmse(
                        common_prediction,
                        candidate_target,
                    )
                    reference_target_candidate_support = rmse(
                        candidate_prediction,
                        reference_target,
                    )
                    reference_value = rmse(
                        common_prediction,
                        reference_target,
                    )
                    for name, observed in (
                        ("own_rmse", own_value),
                        ("reference_rmse", reference_value),
                    ):
                        stored = float(metrics[name])
                        if abs(stored - observed) > 1e-8 * (
                            1.0 + abs(observed)
                        ):
                            raise RuntimeError(
                                f"Stored stability {name} is inconsistent: "
                                f"{path}"
                            )
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_seed": 20260724,
                            "model_seed": seed,
                            "cap": cap,
                            "objective": objective,
                            "selection_metric": selection_metric,
                            "selection_value": (
                                own_value
                                if selection_metric == "own_rmse"
                                else reference_value
                            ),
                            "own_rmse": own_value,
                            "candidate_target_common_support_rmse": (
                                candidate_target_common_support
                            ),
                            "candidate_support_reference_rmse": (
                                reference_target_candidate_support
                            ),
                            "reference_rmse": reference_value,
                            "candidate_target_candidate_support_mse": mse(
                                candidate_prediction, candidate_target
                            ),
                            "candidate_target_common_support_mse": mse(
                                common_prediction, candidate_target
                            ),
                            "reference_target_candidate_support_mse": mse(
                                candidate_prediction, reference_target
                            ),
                            "reference_target_common_support_mse": mse(
                                common_prediction, reference_target
                            ),
                            "result_path": str(path),
                        }
                    )
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Incomplete stability matrix: missing {len(missing)} files; "
            f"first is {missing[0]}"
        )
    return rows


def cap_summary(rows, additional_rows, allow_incomplete=False):
    output = []
    for dataset in DATASETS:
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            for cap in CAPS:
                primary_values = [
                    float(row[metric])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["cap"] == cap
                    and row["split_seed"] == 20260724
                    and row["model_seed"] == 42
                ]
                primary_values.extend(
                    float(row[metric])
                    for row in additional_rows
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["cap"] == cap
                    and row["split_seed"] == 20260724
                    and row["model_seed"] in (43, 44)
                )
                secondary_values = [
                    float(row[metric])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["cap"] == cap
                    and row["split_seed"] in (20260725, 20260726)
                    and row["model_seed"] == 42
                ]
                if not primary_values and not secondary_values:
                    continue
                complete = (
                    len(primary_values) == 3
                    and len(secondary_values) == 2
                )
                if not complete and not allow_incomplete:
                    raise RuntimeError(
                        "Selection aggregate lacks the required primary "
                        f"seeds/secondary splits for {dataset} {objective} "
                        f"cap={cap}: primary={primary_values}, "
                        f"secondary={secondary_values}"
                    )
                if complete:
                    primary_mean = float(np.mean(primary_values))
                    split_values = [primary_mean, *secondary_values]
                    aggregation_rule = (
                        "equal_split_weight_primary_seed_mean_plus_two_"
                        "secondary_seed42"
                    )
                else:
                    provisional = [
                        float(row[metric])
                        for row in rows
                        if row["dataset"] == dataset
                        and row["objective"] == objective
                        and row["cap"] == cap
                        and row["model_seed"] == 42
                    ]
                    if not provisional:
                        continue
                    primary_mean = provisional[0]
                    split_values = provisional
                    aggregation_rule = "provisional_seed42_only"
                output.append(
                    {
                        "dataset": dataset,
                        "objective": objective,
                        "metric": metric,
                        "cap": cap,
                        "mean_across_splits": float(
                            np.mean(split_values)
                        ),
                        "std_across_splits": (
                            float(np.std(split_values, ddof=1))
                            if len(split_values) > 1
                            else 0.0
                        ),
                        "primary_model_seed_values": json.dumps(
                            primary_values
                        ),
                        "primary_model_seed_mean": primary_mean,
                        "split_values": json.dumps(split_values),
                        "aggregation_rule": aggregation_rule,
                        "selection_aggregate_complete": complete,
                    }
                )
    return output


def aggregation_sensitivity(rows, additional_rows, allow_incomplete=False):
    """Recompute cap choices under four predeclared weighting schemes."""
    output = []
    schemes = (
        "split_equal",
        "run_equal",
        "primary_three_seed_mean",
        "seed42_three_split_mean",
    )
    for dataset in DATASETS:
        for objective, metric in (
            ("self_scaled", "own_rmse"),
            ("fixed_reference", "reference_rmse"),
        ):
            scores = {scheme: {} for scheme in schemes}
            for cap in CAPS:
                primary = {
                    int(row["model_seed"]): float(row[metric])
                    for row in [*rows, *additional_rows]
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["cap"] == cap
                    and row["split_seed"] == 20260724
                    and row["model_seed"] in (42, 43, 44)
                }
                secondary = {
                    int(row["split_seed"]): float(row[metric])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["cap"] == cap
                    and row["split_seed"] in (20260725, 20260726)
                    and row["model_seed"] == 42
                }
                complete = (
                    set(primary) == {42, 43, 44}
                    and set(secondary) == {20260725, 20260726}
                )
                if not complete:
                    if allow_incomplete:
                        continue
                    raise RuntimeError(
                        "Aggregation sensitivity lacks the required five "
                        f"runs for {dataset} {objective} cap={cap}"
                    )
                primary_values = [primary[seed] for seed in (42, 43, 44)]
                seed42_split_values = [
                    primary[42],
                    secondary[20260725],
                    secondary[20260726],
                ]
                scores["split_equal"][cap] = float(
                    np.mean(
                        [
                            np.mean(primary_values),
                            secondary[20260725],
                            secondary[20260726],
                        ]
                    )
                )
                scores["run_equal"][cap] = float(
                    np.mean([*primary_values, *secondary.values()])
                )
                scores["primary_three_seed_mean"][cap] = float(
                    np.mean(primary_values)
                )
                scores["seed42_three_split_mean"][cap] = float(
                    np.mean(seed42_split_values)
                )
            for scheme in schemes:
                if len(scores[scheme]) != len(CAPS):
                    if allow_incomplete:
                        continue
                    raise RuntimeError(
                        f"Incomplete {scheme} sensitivity for "
                        f"{dataset} {objective}"
                    )
                selected_cap, selected_value = min(
                    scores[scheme].items(),
                    key=lambda item: (item[1], -item[0]),
                )
                output.append(
                    {
                        "dataset": dataset,
                        "objective": objective,
                        "aggregation_scheme": scheme,
                        "selected_cap": selected_cap,
                        "selection_value": selected_value,
                        "cap_values": json.dumps(
                            {
                                str(cap): scores[scheme][cap]
                                for cap in CAPS
                            },
                            sort_keys=True,
                        ),
                    }
                )
    return output


def decomposition_summary(rows, additional_rows, allow_incomplete=False):
    """Aggregate the 2x2 target/output-ceiling score matrix on identical checkpoints."""
    metrics = (
        (
            "candidate",
            "candidate",
            "candidate_target_candidate_support_mse",
        ),
        (
            "candidate",
            "reference",
            "candidate_target_common_support_mse",
        ),
        (
            "reference",
            "candidate",
            "reference_target_candidate_support_mse",
        ),
        (
            "reference",
            "reference",
            "reference_target_common_support_mse",
        ),
    )
    output = []
    for dataset in DATASETS:
        for checkpoint_objective in ("self_scaled", "fixed_reference"):
            for cap in CAPS:
                for target_scale, output_support, metric in metrics:
                    primary_values = [
                        float(row[metric])
                        for row in rows
                        if row["dataset"] == dataset
                        and row["objective"] == checkpoint_objective
                        and row["cap"] == cap
                        and row["split_seed"] == 20260724
                        and row["model_seed"] == 42
                    ]
                    primary_values.extend(
                        float(row[metric])
                        for row in additional_rows
                        if row["dataset"] == dataset
                        and row["objective"] == checkpoint_objective
                        and row["cap"] == cap
                        and row["split_seed"] == 20260724
                        and row["model_seed"] in (43, 44)
                    )
                    secondary_values = [
                        float(row[metric])
                        for row in rows
                        if row["dataset"] == dataset
                        and row["objective"] == checkpoint_objective
                        and row["cap"] == cap
                        and row["split_seed"] in (20260725, 20260726)
                        and row["model_seed"] == 42
                    ]
                    complete = (
                        len(primary_values) == 3
                        and len(secondary_values) == 2
                    )
                    if not complete and not allow_incomplete:
                        raise RuntimeError(
                            "Incomplete 2x2 decomposition for "
                            f"{dataset} {checkpoint_objective} cap={cap} "
                            f"metric={metric}"
                        )
                    if complete:
                        split_values = [
                            float(np.mean(primary_values)),
                            *secondary_values,
                        ]
                    else:
                        split_values = [
                            float(row[metric])
                            for row in rows
                            if row["dataset"] == dataset
                            and row["objective"] == checkpoint_objective
                            and row["cap"] == cap
                            and row["model_seed"] == 42
                        ]
                    if not split_values:
                        continue
                    output.append(
                        {
                            "dataset": dataset,
                            "checkpoint_objective": checkpoint_objective,
                            "cap": cap,
                            "target_scale": target_scale,
                            "output_support": output_support,
                            "metric": metric,
                            "mean_across_splits": float(
                                np.mean(split_values)
                            ),
                            "std_across_splits": (
                                float(np.std(split_values, ddof=1))
                                if len(split_values) > 1
                                else 0.0
                            ),
                            "split_values": json.dumps(split_values),
                            "selection_aggregate_complete": complete,
                        }
                    )
    return output


def decision_rows(root: Path, cap_rows, allow_incomplete=False):
    path = root / "final_cap_decisions.json"
    if not path.exists():
        if not allow_incomplete:
            raise RuntimeError("Final seed-stable cap lock is missing")
        return []
    lock = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for dataset in DATASETS:
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            item = lock["decisions"][dataset][objective]
            candidates = [
                row
                for row in cap_rows
                if row["dataset"] == dataset
                and (
                    row["objective"] == objective
                    or (
                        objective == "fixed_cap_125"
                        and row["objective"] == "fixed_reference"
                        and row["cap"] == 125
                    )
                )
            ]
            if objective != "fixed_cap_125":
                if len(candidates) != len(CAPS):
                    raise RuntimeError(
                        f"Cannot recompute decision for {dataset} {objective}"
                    )
                recomputed = min(
                    candidates,
                    key=lambda row: (
                        row["mean_across_splits"],
                        -row["cap"],
                    ),
                )
            else:
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Cannot recompute fixed-cap control for {dataset}"
                    )
                recomputed = candidates[0]
            if (
                int(item["selected_cap"]) != int(recomputed["cap"])
                or abs(
                    float(item["selection_value"])
                    - float(recomputed["mean_across_splits"])
                )
                > 1e-10
                * (
                    1.0
                    + abs(float(recomputed["mean_across_splits"]))
                )
            ):
                raise RuntimeError(
                    "Final cap lock differs from independently recomputed "
                    f"selection for {dataset} {objective}"
                )
            output.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "selected_cap": item["selected_cap"],
                    "selection_value": item["selection_value"],
                    "recomputed_selected_cap": recomputed["cap"],
                    "recomputed_selection_value": recomputed[
                        "mean_across_splits"
                    ],
                    "selection_recomputed": True,
                    "lock_protocol": lock["protocol"],
                    "lock_path": str(path),
                }
            )
    return output


def tpe_rows(root: Path, allow_incomplete=False):
    output = []
    missing = []
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            path = (
                root
                / "tpe_locks"
                / f"{dataset}_{objective}.json"
            )
            if not path.exists():
                missing.append(str(path))
                continue
            lock = json.loads(path.read_text(encoding="utf-8"))
            complete_trials = [
                trial
                for trial in lock.get("trial_table", [])
                if trial.get("state") == "COMPLETE"
            ]
            if (
                len(lock.get("trial_table", [])) != 12
                or len(complete_trials) != 12
                or int(lock.get("complete_trials", -1)) != 12
            ):
                raise RuntimeError(
                    f"TPE lock does not contain exactly 12 complete trials: "
                    f"{path}"
                )
            recomputed_best = min(
                complete_trials,
                key=lambda trial: (
                    float(trial["value"]),
                    int(trial["number"]),
                ),
            )
            if (
                int(lock["best_trial"])
                != int(recomputed_best["number"])
                or abs(
                    float(lock["best_value"])
                    - float(recomputed_best["value"])
                )
                > 1e-10
                * (1.0 + abs(float(recomputed_best["value"])))
                or lock["best_params"] != recomputed_best["params"]
                or lock["best_result_path"]
                != recomputed_best["user_attrs"]["result_path"]
            ):
                raise RuntimeError(
                    f"TPE argmin does not reproduce the lock: {path}"
                )
            best_attrs = recomputed_best.get("user_attrs", {})
            best_own_rmse = float(best_attrs.get("own_rmse", np.nan))
            best_reference_rmse = float(
                best_attrs.get("reference_rmse", np.nan)
            )
            if not np.isfinite(
                [best_own_rmse, best_reference_rmse]
            ).all():
                raise RuntimeError(
                    f"TPE winner lacks both validation scales: {path}"
                )
            output.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "trials": lock["complete_trials"],
                    "best_trial": lock["best_trial"],
                    "best_value": lock["best_value"],
                    "recomputed_best_trial": recomputed_best["number"],
                    "recomputed_best_value": recomputed_best["value"],
                    "best_own_rmse": best_own_rmse,
                    "best_reference_rmse": best_reference_rmse,
                    "selection_recomputed": True,
                    **{
                        f"best_{key}": value
                        for key, value in lock["best_params"].items()
                    },
                    "lock_path": str(path),
                }
            )
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Incomplete TPE lock matrix: missing {len(missing)} files; "
            f"first is {missing[0]}"
        )
    return output


def tpe_robustness_rows(root: Path, tpe, allow_incomplete=False):
    """Expose every frozen-winner robustness value without reselection."""
    path = root / "tpe_robustness_summary.json"
    if not path.exists():
        if allow_incomplete:
            return []
        raise RuntimeError("Frozen TPE robustness summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("selection_frozen") is not True
        or summary.get("selection_changed") is not False
        or summary.get("test_evaluated") is not False
    ):
        raise RuntimeError("Frozen TPE robustness summary is invalid")
    lock_index = {
        (row["dataset"], row["objective"]): row for row in tpe
    }
    output = []
    for dataset in ("FD001", "FD002"):
        for objective in (
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        ):
            key = f"{dataset}_{objective}"
            entries = summary.get("metric_summary", {}).get(key)
            paths = summary.get("result_paths", {}).get(key)
            if not isinstance(entries, list) or len(entries) != 3:
                raise RuntimeError(
                    f"Wrong TPE robustness metric matrix for {key}"
                )
            if not isinstance(paths, list) or len(paths) != 3:
                raise RuntimeError(
                    f"Wrong TPE robustness result matrix for {key}"
                )
            by_split = {
                (
                    int(entry["split_seed"]),
                    tuple(int(seed) for seed in entry["model_seeds"]),
                ): entry
                for entry in entries
            }
            expected_keys = {
                (20260724, (43, 44)),
                (20260725, (42,)),
                (20260726, (42,)),
            }
            if set(by_split) != expected_keys:
                raise RuntimeError(
                    f"TPE robustness split/seed matrix differs for {key}"
                )
            expected_metric = (
                "own_rmse"
                if objective == "self_scaled"
                else "reference_rmse"
            )
            if any(
                entry.get("selection_metric") != expected_metric
                for entry in entries
            ):
                raise RuntimeError(
                    f"TPE robustness metric differs from lock for {key}"
                )
            primary_values = [
                float(value)
                for value in by_split[
                    (20260724, (43, 44))
                ]["values"]
            ]
            split_25 = [
                float(value)
                for value in by_split[(20260725, (42,))]["values"]
            ]
            split_26 = [
                float(value)
                for value in by_split[(20260726, (42,))]["values"]
            ]
            if (
                len(primary_values) != 2
                or len(split_25) != 1
                or len(split_26) != 1
                or not np.all(
                    np.isfinite(primary_values + split_25 + split_26)
                )
            ):
                raise RuntimeError(
                    f"TPE robustness values are incomplete for {key}"
                )
            lock = lock_index.get((dataset, objective))
            if lock is None:
                raise RuntimeError(f"TPE lock row is missing for {key}")
            output.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "selection_metric": expected_metric,
                    "selection_split_seed": 20260724,
                    "selection_model_seed": 42,
                    "selection_value": float(lock["best_value"]),
                    "primary_seed43_value": primary_values[0],
                    "primary_seed44_value": primary_values[1],
                    "primary_additional_seed_mean": float(
                        np.mean(primary_values)
                    ),
                    "secondary_split20260725_seed42": split_25[0],
                    "secondary_split20260726_seed42": split_26[0],
                    "selection_frozen": True,
                    "selection_changed": False,
                    "result_paths": json.dumps(paths),
                    "robustness_summary_path": str(path),
                }
            )
    return output


def paired_engine_bootstrap(left, right, target, label, resamples=10000):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not (left.shape == right.shape == target.shape):
        raise ValueError("Paired prediction arrays do not have the same shape")
    point = rmse(left, target) - rmse(right, target)
    if len(target) < 2:
        return point, point, point, float(point < 0)
    seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(target), size=(resamples, len(target))
    )
    left_rmse = np.sqrt(
        np.mean((left[indices] - target[indices]) ** 2, axis=1)
    )
    right_rmse = np.sqrt(
        np.mean((right[indices] - target[indices]) ** 2, axis=1)
    )
    differences = left_rmse - right_rmse
    low, high = np.quantile(differences, [0.025, 0.975])
    return (
        float(point),
        float(low),
        float(high),
        float(np.mean(differences < 0)),
    )


def single_engine_bootstrap_rmse(
    prediction, target, label, resamples=2000
):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target arrays do not have the same shape")
    point = rmse(prediction, target)
    if len(target) < 2:
        return point, point
    seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(target), size=(resamples, len(target))
    )
    values = np.sqrt(
        np.mean((prediction[indices] - target[indices]) ** 2, axis=1)
    )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def add_holm_adjustment(rows, field):
    """Attach standard step-down Holm adjusted p-values in place."""
    order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["wilcoxon_pvalue"],
    )
    adjusted = [1.0] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        value = min(
            1.0,
            (total - rank) * rows[index]["wilcoxon_pvalue"],
        )
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted, strict=True):
        row[field] = value


def exact_signed_rank_test(differences, zero_tolerance=1e-12):
    """Two-sided exact sign-flip Wilcoxon statistic, including ties/zeros.

    Zero differences are removed.  Average ranks are used for tied absolute
    differences, and all ``2**m`` sign assignments of the remaining ranks are
    enumerated.  With five paired seeds this is deterministic and avoids
    SciPy-version-dependent fallback from ``method='exact'``.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Signed-rank differences must be finite and one-dimensional")
    nonzero = values[np.abs(values) > zero_tolerance]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0
    ranks = rankdata(np.abs(nonzero), method="average")
    total = float(np.sum(ranks))
    positive = float(np.sum(ranks[nonzero > 0]))
    observed = min(positive, total - positive)
    permutation_statistics = []
    for mask in range(1 << len(nonzero)):
        assigned_positive = float(
            np.sum(
                [
                    ranks[index]
                    for index in range(len(nonzero))
                    if mask & (1 << index)
                ]
            )
        )
        permutation_statistics.append(
            min(assigned_positive, total - assigned_positive)
        )
    pvalue = float(
        np.mean(
            np.asarray(permutation_statistics, dtype=np.float64)
            <= observed + 1e-12
        )
    )
    return float(observed), min(1.0, pvalue), int(len(nonzero))


def final_rows(root: Path):
    path = root / "final_test_summary.json"
    marker = root / "TEST_EVALUATED"
    if not path.exists():
        if marker.exists():
            raise RuntimeError(
                "TEST_EVALUATED exists without final_test_summary.json"
            )
        return [], []
    if not marker.exists():
        raise RuntimeError(
            "Final test summary exists without TEST_EVALUATED marker"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    plan_path = root / "immutable_test_plan.json"
    started_path = root / "TEST_EVALUATION_STARTED.json"
    if not plan_path.is_file() or not started_path.is_file():
        raise RuntimeError(
            "Final summary exists without immutable plan/start marker"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_completed_chain(
        summary, marker_value, plan, root, started_path
    )
    decision_lock_path = root / "final_cap_decisions.json"
    if not decision_lock_path.is_file():
        raise RuntimeError("Final cap decision lock is missing")
    decision_lock = json.loads(
        decision_lock_path.read_text(encoding="utf-8")
    )
    if stable_hash(decision_lock) != plan.get("decision_lock_sha256"):
        raise RuntimeError(
            "Final cap decision lock differs from immutable test plan"
        )
    if set(summary.get("selected_results", {})) != set(DATASETS):
        raise RuntimeError(
            "Final selected-result datasets differ from FD001--FD004"
        )
    rows = []
    pair_rows = []
    bound_result_paths = {
        str(Path(value).resolve())
        for value in summary.get("result_file_sha256", {})
    }

    def verified_selected_test(
        item,
        objective_key,
        expected_cap,
        expected_dataset,
        spec,
        expected_tpe_objective=None,
    ):
        result_path = Path(item["result_path"]).resolve()
        planned_path = (
            root / "results" / spec["result_filename"]
        ).resolve()
        if str(result_path) not in bound_result_paths:
            raise RuntimeError(
                f"Selected result is outside the final hash set: {result_path}"
            )
        if result_path != planned_path:
            raise RuntimeError(
                f"Selected result does not match its planned arm: {result_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("protocol") != "confirmatory-rul-cap-v2"
            or result.get("dataset") != expected_dataset
            or result.get("campaign_id") != spec["campaign_id"]
            or result.get("tag") != spec["tag"]
            or int(result["train_cap"]) != int(expected_cap)
            or float(result["reporting_cap"])
            != float(spec["reporting_cap"])
            or result.get("split_seed") != spec["split_seed"]
            or result.get("endpoint_seed") != spec["endpoint_seed"]
            or result.get("model_seeds") != spec["model_seeds"]
            or not config_matches(result.get("config"), spec["config"])
            or result.get("validation_only") is not False
            or result.get("test_evaluated") is not True
            or item["test"] != result["test_results"][objective_key]
        ):
            raise RuntimeError(
                f"Selected summary payload differs from result: {result_path}"
            )
        if (
            expected_tpe_objective is not None
            and spec.get("objective") != expected_tpe_objective
        ):
            raise RuntimeError(
                f"TPE arm identity differs from plan: {result_path}"
            )
        return item["test"]

    for dataset in DATASETS:
        methods = summary["selected_results"][dataset]
        expected_method_keys = {
            "self_scaled",
            "fixed_reference",
            "fixed_cap_125",
        }
        if dataset in {"FD001", "FD002"}:
            expected_method_keys.add("tpe")
        if set(methods) != expected_method_keys:
            raise RuntimeError(
                f"Unexpected final method set for {dataset}: {sorted(methods)}"
            )
        method_caps = {
            "cap_grid_self_scaled": methods["self_scaled"][
                "selected_cap"
            ],
            "cap_grid_fixed_reference": methods["fixed_reference"][
                "selected_cap"
            ],
            "fixed_cap_125": 125,
        }
        for objective in ("self_scaled", "fixed_reference"):
            locked_cap = int(
                decision_lock["decisions"][dataset][objective][
                    "selected_cap"
                ]
            )
            if int(method_caps[f"cap_grid_{objective}"]) != locked_cap:
                raise RuntimeError(
                    f"Selected cap differs from decision lock for "
                    f"{dataset} {objective}"
                )
        method_items = {
            "cap_grid_self_scaled": verified_selected_test(
                methods["self_scaled"],
                "self_scaled",
                method_caps["cap_grid_self_scaled"],
                dataset,
                plan["cap_result_specs"][dataset][
                    str(method_caps["cap_grid_self_scaled"])
                ],
            ),
            "cap_grid_fixed_reference": verified_selected_test(
                methods["fixed_reference"],
                "fixed_reference",
                method_caps["cap_grid_fixed_reference"],
                dataset,
                plan["cap_result_specs"][dataset][
                    str(method_caps["cap_grid_fixed_reference"])
                ],
            ),
            "fixed_cap_125": verified_selected_test(
                methods["fixed_cap_125"],
                "fixed_reference",
                125,
                dataset,
                plan["cap_result_specs"][dataset]["125"],
            ),
        }
        if "tpe" in methods:
            method_caps.update(
                {
                    f"tpe_{objective}": methods["tpe"][objective][
                        "best_params"
                    ]["train_cap"]
                    for objective in (
                        "self_scaled",
                        "fixed_reference",
                        "fixed_cap_125",
                    )
                }
            )
            for objective in (
                "self_scaled",
                "fixed_reference",
                "fixed_cap_125",
            ):
                objective_key = (
                    "self_scaled"
                    if objective == "self_scaled"
                    else "fixed_reference"
                )
                tpe_key = f"{dataset}_{objective}"
                spec = plan["tpe_result_specs"][tpe_key]
                if (
                    methods["tpe"][objective]["best_params"]
                    != plan["tpe_locks"][tpe_key]["best_params"]
                ):
                    raise RuntimeError(
                        f"TPE parameters differ from immutable plan for "
                        f"{tpe_key}"
                    )
                method_items[f"tpe_{objective}"] = verified_selected_test(
                    methods["tpe"][objective],
                    objective_key,
                    method_caps[f"tpe_{objective}"],
                    dataset,
                    spec,
                    expected_tpe_objective=objective,
                )
        for method, test in method_items.items():
            ensemble = test["ensemble"]
            train_cap = int(method_caps[method])
            original_target = np.asarray(
                ensemble["original_targets"], dtype=np.float64
            )
            candidate_prediction = np.asarray(
                ensemble["candidate_support_predictions"],
                dtype=np.float64,
            )
            common_prediction = np.asarray(
                ensemble["common_support_predictions"],
                dtype=np.float64,
            )
            units = ensemble["units"]
            if (
                len(units) != len(set(units))
                or not (
                    len(units)
                    == len(original_target)
                    == len(candidate_prediction)
                    == len(common_prediction)
                )
                or not np.all(
                    np.isfinite(
                        np.concatenate(
                            [
                                original_target,
                                candidate_prediction,
                                common_prediction,
                            ]
                        )
                    )
                )
                or np.any(candidate_prediction < -1e-8)
                or np.any(candidate_prediction > train_cap + 1e-8)
                or np.any(common_prediction < -1e-8)
                or np.any(common_prediction > 125.0 + 1e-8)
            ):
                raise RuntimeError(
                    f"Invalid locked ensemble arrays for {dataset} {method}"
                )
            own_target = np.clip(original_target, 0.0, train_cap)
            reference_target = np.clip(
                original_target, 0.0, 125.0
            )
            own_rmse_ensemble = rmse(
                candidate_prediction, own_target
            )
            candidate_target_common_support_rmse = rmse(
                common_prediction, own_target
            )
            reference_rmse_ensemble = rmse(
                common_prediction, reference_target
            )
            candidate_reference_rmse = rmse(
                candidate_prediction, reference_target
            )
            candidate_original_rmse = rmse(
                candidate_prediction, original_target
            )
            original_rmse_ensemble = rmse(
                common_prediction, original_target
            )
            nasa_reference = nasa_score(
                common_prediction, reference_target
            )
            reference_decisions = reference_decision_metrics(
                common_prediction, reference_target
            )
            stored_checks = {
                "own_rmse": own_rmse_ensemble,
                "reference_rmse": reference_rmse_ensemble,
                "original_rmse": original_rmse_ensemble,
                "nasa_score_reference": nasa_reference,
            }
            for key, recomputed in stored_checks.items():
                stored = float(ensemble[key])
                if abs(stored - recomputed) > 1e-8 * (
                    1.0 + abs(recomputed)
                ):
                    raise RuntimeError(
                        "Stored final metric is inconsistent for "
                        f"{dataset} {method} {key}"
                    )
            per_seed = test.get("per_seed", [])
            seed_ids = {int(item["seed"]) for item in per_seed}
            if len(per_seed) != 5 or seed_ids != FINAL_MODEL_SEEDS:
                raise RuntimeError(
                    f"Final result for {dataset} {method} does not contain "
                    "exactly model seeds 45--49"
                )
            seed_aggregates = {}
            for metric_name in ("own_rmse", "reference_rmse"):
                values = np.asarray(
                    [float(item[metric_name]) for item in per_seed],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"Non-finite per-seed {metric_name} for "
                        f"{dataset} {method}"
                    )
                mean_value = float(np.mean(values))
                std_value = float(np.std(values, ddof=1))
                stored_summary = test["summary"][metric_name]
                stored_values = np.asarray(
                    stored_summary.get("values", []),
                    dtype=np.float64,
                )
                if (
                    abs(float(stored_summary["mean"]) - mean_value)
                    > 1e-10 * (1.0 + abs(mean_value))
                    or abs(float(stored_summary["std"]) - std_value)
                    > 1e-10 * (1.0 + abs(std_value))
                    or stored_values.shape != values.shape
                    or not np.allclose(
                        stored_values,
                        values,
                        rtol=1e-10,
                        atol=1e-12,
                    )
                ):
                    raise RuntimeError(
                        f"Stored per-seed aggregate is inconsistent for "
                        f"{dataset} {method} {metric_name}"
                    )
                seed_aggregates[metric_name] = (mean_value, std_value)
            recomputed_saturation = float(
                np.mean(original_target > train_cap)
            )
            recomputed_exposure = float(
                np.sqrt(
                    np.mean(np.maximum(original_target - train_cap, 0.0) ** 2)
                )
            )
            recomputed_reference_exposure = float(
                np.sqrt(
                    np.mean(
                        np.maximum(reference_target - train_cap, 0.0) ** 2
                    )
                )
            )
            if (
                abs(
                    float(ensemble["saturation_fraction"])
                    - recomputed_saturation
                )
                > 1e-12
                or abs(
                    float(ensemble["exposure_bound"])
                    - recomputed_exposure
                )
                > 1e-8 * (1.0 + abs(recomputed_exposure))
            ):
                raise RuntimeError(
                    f"Stored final saturation/exposure is inconsistent for "
                    f"{dataset} {method}"
                )
            candidate_hidden_mse_gap = (
                candidate_original_rmse**2 - own_rmse_ensemble**2
            )
            exposure_bound_squared = recomputed_exposure**2
            if candidate_hidden_mse_gap < exposure_bound_squared - 1e-8 * (
                1.0
                + abs(candidate_hidden_mse_gap)
                + abs(exposure_bound_squared)
            ):
                raise RuntimeError(
                    f"Final candidate-support MSE-gap bound failed for "
                    f"{dataset} {method}"
                )
            reference_hidden_mse_gap = (
                candidate_reference_rmse**2 - own_rmse_ensemble**2
            )
            reference_exposure_bound_squared = (
                recomputed_reference_exposure**2
            )
            if (
                reference_hidden_mse_gap
                < reference_exposure_bound_squared
                - 1e-8
                * (
                    1.0
                    + abs(reference_hidden_mse_gap)
                    + abs(reference_exposure_bound_squared)
                )
            ):
                raise RuntimeError(
                    f"Final reference-scale MSE-gap bound failed for "
                    f"{dataset} {method}"
                )
            bootstrap_low, bootstrap_high = single_engine_bootstrap_rmse(
                common_prediction,
                reference_target,
                f"{dataset}:{method}:individual-reference-rmse",
                resamples=2000,
            )
            row = {
                "dataset": dataset,
                "method": method,
                "train_cap": train_cap,
                "own_rmse_mean": seed_aggregates["own_rmse"][0],
                "own_rmse_std": seed_aggregates["own_rmse"][1],
                "own_rmse_ensemble": own_rmse_ensemble,
                "candidate_target_common_support_rmse_ensemble": (
                    candidate_target_common_support_rmse
                ),
                "reference_rmse_mean": seed_aggregates[
                    "reference_rmse"
                ][0],
                "reference_rmse_std": seed_aggregates[
                    "reference_rmse"
                ][1],
                "reference_rmse_ensemble": reference_rmse_ensemble,
                "candidate_support_reference_rmse_ensemble": (
                    candidate_reference_rmse
                ),
                "candidate_support_original_rmse_ensemble": (
                    candidate_original_rmse
                ),
                "candidate_hidden_mse_gap": candidate_hidden_mse_gap,
                "exposure_bound_squared": exposure_bound_squared,
                "bound_fraction_of_hidden_gap": (
                    exposure_bound_squared / candidate_hidden_mse_gap
                    if candidate_hidden_mse_gap > 0
                    else None
                ),
                "bound_satisfied": True,
                "saturation_fraction": recomputed_saturation,
                "exposure_bound": recomputed_exposure,
                "reference_hidden_mse_gap": reference_hidden_mse_gap,
                "reference_exposure_bound_squared": (
                    reference_exposure_bound_squared
                ),
                "reference_bound_fraction_of_hidden_gap": (
                    reference_exposure_bound_squared
                    / reference_hidden_mse_gap
                    if reference_hidden_mse_gap > 0
                    else None
                ),
                "reference_exposure_bound": (
                    recomputed_reference_exposure
                ),
                "reference_rmse_bootstrap_low": bootstrap_low,
                "reference_rmse_bootstrap_high": bootstrap_high,
                "original_rmse_ensemble": original_rmse_ensemble,
                "nasa_score_reference_ensemble": nasa_reference,
                "mean_error_ensemble": reference_decisions[
                    "mean_error"
                ],
                "late_prediction_fraction": reference_decisions[
                    "late_prediction_fraction"
                ],
            }
            for threshold in (20, 30, 50):
                row[f"n_due_at_{threshold}"] = reference_decisions[
                    f"n_due_at_{threshold}"
                ]
                row[f"n_not_due_at_{threshold}"] = reference_decisions[
                    f"n_not_due_at_{threshold}"
                ]
                row[f"delayed_action_rate_at_{threshold}"] = (
                    reference_decisions[
                    f"delayed_action_rate_at_{threshold}"
                    ]
                )
                row[f"premature_action_rate_at_{threshold}"] = (
                    reference_decisions[
                    f"premature_action_rate_at_{threshold}"
                    ]
                )
            rows.append(row)

        comparisons = [
            (
                "fixed_vs_self_cap_grid",
                method_items["cap_grid_fixed_reference"],
                method_items["cap_grid_self_scaled"],
            ),
            (
                "fixed_cap_grid_vs_fixed125",
                method_items["cap_grid_fixed_reference"],
                method_items["fixed_cap_125"],
            ),
        ]
        if "tpe_fixed_reference" in method_items:
            comparisons.append(
                (
                    "fixed_vs_self_tpe",
                    method_items["tpe_fixed_reference"],
                    method_items["tpe_self_scaled"],
                )
            )
            comparisons.append(
                (
                    "fixed_reference_tpe_vs_fixed_cap_125_tpe",
                    method_items["tpe_fixed_reference"],
                    method_items["tpe_fixed_cap_125"],
                )
            )
        for label, left, right in comparisons:
            left_seed_map = {}
            right_seed_map = {}
            for item in left["per_seed"]:
                seed = int(item["seed"])
                if seed in left_seed_map:
                    raise ValueError(f"Duplicate left seed {seed}")
                left_seed_map[seed] = float(item["reference_rmse"])
            for item in right["per_seed"]:
                seed = int(item["seed"])
                if seed in right_seed_map:
                    raise ValueError(f"Duplicate right seed {seed}")
                right_seed_map[seed] = float(item["reference_rmse"])
            if (
                set(left_seed_map) != set(right_seed_map)
                or set(left_seed_map) != FINAL_MODEL_SEEDS
                or not all(
                    math.isfinite(value)
                    for value in [
                        *left_seed_map.values(),
                        *right_seed_map.values(),
                    ]
                )
            ):
                raise ValueError(
                    f"Expected paired model seeds 45--49 for "
                    f"{dataset} {label}"
                )
            paired_seeds = sorted(left_seed_map)
            left_values = np.asarray(
                [left_seed_map[seed] for seed in paired_seeds]
            )
            right_values = np.asarray(
                [right_seed_map[seed] for seed in paired_seeds]
            )
            differences = left_values - right_values
            statistic, pvalue, nonzero_count = exact_signed_rank_test(
                differences
            )
            left_ensemble = left["ensemble"]
            right_ensemble = right["ensemble"]
            if (
                left_ensemble["units"] != right_ensemble["units"]
                or left_ensemble["original_targets"]
                != right_ensemble["original_targets"]
            ):
                raise ValueError(
                    f"Unpaired test endpoints for {dataset} {label}"
                )
            reference_target = np.clip(
                np.asarray(
                    left_ensemble["original_targets"],
                    dtype=np.float64,
                ),
                0.0,
                125.0,
            )
            (
                engine_difference,
                engine_low,
                engine_high,
                fraction_negative,
            ) = paired_engine_bootstrap(
                left_ensemble["common_support_predictions"],
                right_ensemble["common_support_predictions"],
                reference_target,
                f"{dataset}:{label}",
            )
            pair_rows.append(
                {
                    "dataset": dataset,
                    "comparison": label,
                    "paired_difference_values": json.dumps(
                        differences.tolist()
                    ),
                    "paired_model_seeds": json.dumps(paired_seeds),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(
                        np.median(differences)
                    ),
                    "left_better_seed_count": int(
                        np.sum(differences < 0)
                    ),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_pvalue": pvalue,
                    "wilcoxon_nonzero_pairs": nonzero_count,
                    "wilcoxon_method": (
                        "exact_sign_flip_average_ranks_zero_dropped"
                    ),
                    "n_model_seeds": len(differences),
                    "paired_engine_rmse_difference": engine_difference,
                    "paired_engine_bootstrap_95ci_low": engine_low,
                    "paired_engine_bootstrap_95ci_high": engine_high,
                    "bootstrap_fraction_negative": (
                        fraction_negative
                    ),
                    "n_test_engines": len(reference_target),
                }
            )
    if pair_rows:
        expected_pairs = {
            *( (dataset, "fixed_vs_self_cap_grid") for dataset in DATASETS ),
            *( (dataset, "fixed_cap_grid_vs_fixed125") for dataset in DATASETS ),
            ("FD001", "fixed_vs_self_tpe"),
            ("FD001", "fixed_reference_tpe_vs_fixed_cap_125_tpe"),
            ("FD002", "fixed_vs_self_tpe"),
            ("FD002", "fixed_reference_tpe_vs_fixed_cap_125_tpe"),
        }
        observed_pairs = {
            (row["dataset"], row["comparison"]) for row in pair_rows
        }
        if len(pair_rows) != 12 or observed_pairs != expected_pairs:
            raise RuntimeError(
                "The all-contrast Holm family must contain exactly the "
                "12 predeclared dataset/comparison pairs"
            )

        add_holm_adjustment(
            pair_rows, "wilcoxon_holm_all12_pvalue"
        )
        primary = [
            row
            for row in pair_rows
            if row["comparison"] == "fixed_vs_self_cap_grid"
        ]
        if len(primary) != 4:
            raise RuntimeError(
                "Primary Holm family must contain four cap-grid contrasts"
            )
        if {row["dataset"] for row in primary} != set(DATASETS):
            raise RuntimeError(
                "Primary Holm family must cover FD001--FD004 exactly"
            )
        add_holm_adjustment(
            primary, "wilcoxon_holm_primary_pvalue"
        )
        secondary = [
            row
            for row in pair_rows
            if row["comparison"] != "fixed_vs_self_cap_grid"
        ]
        if len(secondary) != 8:
            raise RuntimeError(
                "Secondary Holm family must contain eight contrasts"
            )
        add_holm_adjustment(
            secondary, "wilcoxon_holm_secondary_pvalue"
        )
        for row in pair_rows:
            row.setdefault("wilcoxon_holm_primary_pvalue", None)
            row.setdefault("wilcoxon_holm_secondary_pvalue", None)
    return rows, pair_rows


def latex_fragment(decisions, finals):
    by_decision = {
        (row["dataset"], row["objective"]): row for row in decisions
    }
    lines = [
        "% Generated by summarize_confirmatory.py; do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Caps locked before standard test-split access.}",
        "\\label{tab:confirmatory-caps}",
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Subset & Self-scaled & Fixed-reference \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        if (dataset, "self_scaled") not in by_decision:
            continue
        lines.append(
            f"{dataset} & "
            f"{by_decision[(dataset, 'self_scaled')]['selected_cap']} & "
            f"{by_decision[(dataset, 'fixed_reference')]['selected_cap']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    if finals:
        indexed = {
            (row["dataset"], row["method"]): row for row in finals
        }
        lines.extend(
            [
                "",
                "\\begin{table*}[t]",
                "\\centering",
                "\\caption{Locked five-seed test results on the common "
                "reference scale.}",
                "\\label{tab:confirmatory-test}",
                "\\begin{tabular}{@{}lrrr@{}}",
                "\\toprule",
                "Subset & Self-scaled & Fixed-reference & Fixed $c=125$ \\\\",
                "\\midrule",
            ]
        )
        for dataset in DATASETS:
            keys = [
                (dataset, "cap_grid_self_scaled"),
                (dataset, "cap_grid_fixed_reference"),
                (dataset, "fixed_cap_125"),
            ]
            if not all(key in indexed for key in keys):
                continue
            formatted = []
            for key in keys:
                row = indexed[key]
                formatted.append(
                    f"{row['reference_rmse_mean']:.2f}"
                    f"$\\pm${row['reference_rmse_std']:.2f}"
                )
            lines.append(
                f"{dataset} & {formatted[0]} & {formatted[1]} & "
                f"{formatted[2]} \\\\"
            )
        lines.extend(
            ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"E:\NASA_Lab"),
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Permit partial monitoring summaries before locks/tests exist.",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not args.allow_incomplete:
        run_preflight(root, args.data_root)
    destination = root / "summaries"
    validation, bounds = screening_rows(
        root, allow_incomplete=args.allow_incomplete
    )
    stability = stability_rows(
        root, allow_incomplete=args.allow_incomplete
    )
    cap_rows = cap_summary(
        validation,
        stability,
        allow_incomplete=args.allow_incomplete,
    )
    aggregation_rows = aggregation_sensitivity(
        validation,
        stability,
        allow_incomplete=args.allow_incomplete,
    )
    decomposition = decomposition_summary(
        validation,
        stability,
        allow_incomplete=args.allow_incomplete,
    )
    decisions = decision_rows(
        root,
        cap_rows,
        allow_incomplete=args.allow_incomplete,
    )
    tpe = tpe_rows(root, allow_incomplete=args.allow_incomplete)
    tpe_robustness = tpe_robustness_rows(
        root, tpe, allow_incomplete=args.allow_incomplete
    )
    finals, pairs = final_rows(root)
    write_csv(destination / "validation_long.csv", validation)
    write_csv(destination / "stability_long.csv", stability)
    write_csv(destination / "exposure_bound_audit.csv", bounds)
    write_csv(destination / "cap_summary.csv", cap_rows)
    write_csv(
        destination / "aggregation_sensitivity.csv",
        aggregation_rows,
    )
    write_csv(
        destination / "target_support_decomposition.csv",
        decomposition,
    )
    write_csv(destination / "locked_decisions.csv", decisions)
    write_csv(destination / "tpe_locks.csv", tpe)
    write_csv(
        destination / "tpe_robustness.csv", tpe_robustness
    )
    write_csv(destination / "final_test.csv", finals)
    write_csv(destination / "paired_tests.csv", pairs)
    if bounds and not all(row["bound_satisfied"] for row in bounds):
        raise AssertionError("At least one candidate-support bound failed")
    if not args.allow_incomplete:
        expected_counts = {
            "validation_rows": 120,
            "stability_rows": 80,
            "bound_rows": 120,
            "cap_rows": 40,
            "aggregation_sensitivity_rows": 32,
            "decomposition_rows": 160,
            "decision_rows": 12,
            "tpe_rows": 6,
            "tpe_robustness_rows": 6,
            "final_rows": 18,
            "paired_tests": 12,
        }
        observed_counts = {
            "validation_rows": len(validation),
            "stability_rows": len(stability),
            "bound_rows": len(bounds),
            "cap_rows": len(cap_rows),
            "aggregation_sensitivity_rows": len(aggregation_rows),
            "decomposition_rows": len(decomposition),
            "decision_rows": len(decisions),
            "tpe_rows": len(tpe),
            "tpe_robustness_rows": len(tpe_robustness),
            "final_rows": len(finals),
            "paired_tests": len(pairs),
        }
        if observed_counts != expected_counts:
            raise RuntimeError(
                "Strict summary matrix is incomplete: "
                f"observed={observed_counts}, expected={expected_counts}"
            )
    atomic_text(
        destination / "generated_confirmatory_results.tex",
        latex_fragment(decisions, finals),
    )
    csv_names = [
        "validation_long.csv",
        "stability_long.csv",
        "exposure_bound_audit.csv",
        "cap_summary.csv",
        "aggregation_sensitivity.csv",
        "target_support_decomposition.csv",
        "locked_decisions.csv",
        "tpe_locks.csv",
        "tpe_robustness.csv",
        "final_test.csv",
        "paired_tests.csv",
    ]
    csv_hashes = {
        name: sha256_file(destination / name)
        for name in csv_names
        if (destination / name).is_file()
    }
    manifest_path = root / "campaign_manifest.json"
    status = {
        "protocol": "confirmatory-summary-v2",
        "complete": not args.allow_incomplete,
        "campaign_id": (
            json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "campaign_id"
            )
            if manifest_path.is_file()
            else None
        ),
        "validation_rows": len(validation),
        "stability_rows": len(stability),
        "bound_rows": len(bounds),
        "bound_failures": sum(
            not row["bound_satisfied"] for row in bounds
        ),
        "cap_rows": len(cap_rows),
        "aggregation_sensitivity_rows": len(aggregation_rows),
        "decomposition_rows": len(decomposition),
        "decision_rows": len(decisions),
        "tpe_rows": len(tpe),
        "tpe_robustness_rows": len(tpe_robustness),
        "final_rows": len(finals),
        "paired_tests": len(pairs),
        "csv_sha256": csv_hashes,
        "final_test_summary_sha256": (
            sha256_file(root / "final_test_summary.json")
            if (root / "final_test_summary.json").is_file()
            else None
        ),
        "test_evaluated_marker_sha256": (
            sha256_file(root / "TEST_EVALUATED")
            if (root / "TEST_EVALUATED").is_file()
            else None
        ),
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "manuscript_asset_generator_sha256": sha256_file(
            Path(__file__).resolve().with_name(
                "generate_manuscript_assets.py"
            )
        ),
    }
    atomic_text(
        destination / "summary_status.json",
        json.dumps(status, indent=2) + "\n",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
