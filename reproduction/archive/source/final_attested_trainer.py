"""Attestation and prediction-archive shim for the frozen official trainer.

The imported training implementation remains byte-identical to the
validation campaign. This shim consumes one launch authorization, preserves
the per-seed prediction arrays already computed by the frozen evaluator, and
writes its nonce/hash chain into the official result before the frozen
trainer's atomic JSON commit. It does not perform a second test evaluation.
"""

from __future__ import annotations

import math
import os
import string
import sys

import confirmatory_rul_train as frozen


ENVIRONMENT = {
    "test_plan_sha256": "CCV2_TEST_PLAN_SHA256",
    "start_marker_sha256": "CCV2_START_MARKER_SHA256",
    "launch_authorization_sha256": "CCV2_LAUNCH_SHA256",
    "launch_nonce": "CCV2_LAUNCH_NONCE",
}
ARRAY_FIELDS = (
    "units",
    "raw_predictions",
    "candidate_support_predictions",
    "common_support_predictions",
    "original_targets",
)
SCALAR_FIELDS = (
    "own_rmse",
    "reference_rmse",
    "original_rmse",
    "nasa_score_reference",
    "nasa_score_original",
)
DECISION_FIELDS = (
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


def require_hex(name: str, length: int = 64) -> str:
    value = os.environ.get(name, "").lower()
    if len(value) != length or any(
        character not in string.hexdigits.lower() for character in value
    ):
        raise RuntimeError(f"Missing or malformed official attestation {name}")
    return value


def attach_per_seed_predictions(
    result: dict,
    captured: list[dict],
    train_cap: float,
    reporting_cap: float,
) -> dict:
    """Attach arrays and normalize decision diagnostics to the reference target."""
    offset = 0
    for objective in ("self_scaled", "fixed_reference"):
        rows = result[objective]["per_seed"]
        count = len(rows)
        seed_bundles = captured[offset : offset + count]
        if len(seed_bundles) != count:
            raise RuntimeError("Official per-seed prediction capture is short")
        for row, bundle in zip(rows, seed_bundles, strict=True):
            for field in SCALAR_FIELDS:
                if field not in row or field not in bundle or not math.isclose(
                    float(row[field]),
                    float(bundle[field]),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(
                        "Captured prediction bundle does not match "
                        f"the frozen scalar row for {objective}: {field}"
                    )
            for field in ARRAY_FIELDS:
                if field not in bundle:
                    raise RuntimeError(
                        f"Captured prediction bundle lacks {field}"
                    )
                row[field] = bundle[field]
            reference_target = frozen.np.clip(
                frozen.np.asarray(bundle["original_targets"], dtype=float),
                0.0,
                reporting_cap,
            )
            row.update(
                frozen.decision_metrics(
                    bundle["common_support_predictions"],
                    reference_target,
                )
            )
        for field in DECISION_FIELDS:
            result[objective]["summary"][field] = frozen.aggregate(
                [row[field] for row in rows]
            )
        offset += count
        # The frozen evaluator calls metric_bundle once more for the median
        # ensemble. Its legacy raw field is based on an already clipped
        # common-support ensemble, so reconstruct the true raw median from
        # the preserved per-seed raw arrays before attestation.
        offset += 1
        ensemble = result[objective]["ensemble"]
        raw_ensemble = frozen.np.median(
            frozen.np.asarray(
                [row["raw_predictions"] for row in rows], dtype=float
            ),
            axis=0,
        )
        candidate_ensemble = frozen.np.clip(raw_ensemble, 0.0, train_cap)
        common_ensemble = frozen.np.clip(
            raw_ensemble, 0.0, reporting_cap
        )
        for field, expected in (
            ("candidate_support_predictions", candidate_ensemble),
            ("common_support_predictions", common_ensemble),
        ):
            observed = frozen.np.asarray(ensemble[field], dtype=float)
            if not frozen.np.allclose(
                observed, expected, rtol=1e-10, atol=1e-10
            ):
                raise RuntimeError(
                    f"Frozen median ensemble is inconsistent for {field}"
                )
        ensemble["raw_predictions"] = [
            float(value) for value in raw_ensemble
        ]
        ensemble["candidate_support_predictions"] = [
            float(value) for value in candidate_ensemble
        ]
        ensemble["common_support_predictions"] = [
            float(value) for value in common_ensemble
        ]
        reference_target = frozen.np.clip(
            frozen.np.asarray(ensemble["original_targets"], dtype=float),
            0.0,
            reporting_cap,
        )
        ensemble.update(
            frozen.decision_metrics(
                ensemble["common_support_predictions"],
                reference_target,
            )
        )
        # The formal intervals are regenerated from archived arrays by the
        # content-addressed summarizer.  Removing the trainer-local intervals
        # avoids retaining a second, unplanned bootstrap definition.
        ensemble.pop("reference_rmse_bootstrap_95ci", None)
        ensemble.pop("original_rmse_bootstrap_95ci", None)
    if offset != len(captured):
        raise RuntimeError(
            "Unexpected metric-bundle calls during official evaluation"
        )
    return result


def main() -> None:
    if "--validation_only" in sys.argv:
        raise RuntimeError(
            "The final attestation shim is only for official test evaluation"
        )
    attestation = {
        key: require_hex(environment_name)
        for key, environment_name in ENVIRONMENT.items()
    }
    original_atomic_json = frozen.atomic_json
    original_metric_bundle = frozen.metric_bundle
    original_evaluate_test = frozen.evaluate_test
    commits = 0
    evaluation_calls = 0

    def archived_evaluate_test(*args, **kwargs):
        nonlocal evaluation_calls
        if evaluation_calls:
            raise RuntimeError("Official test evaluator was invoked twice")
        evaluation_calls += 1
        captured = []

        def capturing_metric_bundle(*metric_args, **metric_kwargs):
            bundle = original_metric_bundle(*metric_args, **metric_kwargs)
            captured.append(bundle)
            return bundle

        frozen.metric_bundle = capturing_metric_bundle
        try:
            result = original_evaluate_test(*args, **kwargs)
        finally:
            frozen.metric_bundle = original_metric_bundle
        if not args or not hasattr(args[0], "reporting_cap"):
            raise RuntimeError(
                "Official evaluator call lacks the reporting-cap contract"
            )
        return attach_per_seed_predictions(
            result,
            captured,
            float(args[0].train_cap),
            float(args[0].reporting_cap),
        )

    def attested_atomic_json(path, value):
        nonlocal commits
        if value.get("protocol") != "confirmatory-rul-cap-v2":
            raise RuntimeError("Unexpected result protocol in final shim")
        if (
            value.get("validation_only") is not False
            or value.get("test_evaluated") is not True
        ):
            raise RuntimeError("Final shim refused a non-test result")
        if commits:
            raise RuntimeError("Final shim permits exactly one result commit")
        commits += 1
        enriched = dict(value)
        enriched["official_execution_attestation"] = {
            "protocol": "confirmatory-final-execution-attestation-v2",
            **attestation,
        }
        original_atomic_json(path, enriched)

    frozen.atomic_json = attested_atomic_json
    frozen.evaluate_test = archived_evaluate_test
    try:
        frozen.main()
    finally:
        frozen.atomic_json = original_atomic_json
        frozen.metric_bundle = original_metric_bundle
        frozen.evaluate_test = original_evaluate_test
    if commits != 1 or evaluation_calls != 1:
        raise RuntimeError(
            "Frozen trainer did not produce exactly one attested evaluation"
        )


if __name__ == "__main__":
    main()
