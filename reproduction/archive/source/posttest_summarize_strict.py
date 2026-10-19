"""Strict post-test summarization after the immutable campaign is complete.

The frozen preflight routine is intentionally designed for the pre-test state.
Calling it after the official-test marker exists attempts to rebuild the
pre-marker inventory and therefore rejects a correctly completed campaign.
This wrapper does not modify any frozen source.  It replaces only the in-memory
preflight callback with post-test integrity checks, then runs the original
summarizer in strict (not ``--allow_incomplete``) mode.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_confirmatory_final as final_runner
import summarize_confirmatory as summarizer


ROOT = Path(r"E:\NASA_Lab\ress_confirmatory_v2")
DATA_ROOT = Path(r"E:\NASA_Lab")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def posttest_preflight(root: Path, data_root: Path) -> dict:
    root = root.resolve()
    data_root = data_root.resolve()
    if root != ROOT.resolve() or data_root != DATA_ROOT.resolve():
        raise RuntimeError("Unexpected post-test campaign or data root")

    plan_path = root / "immutable_test_plan.json"
    started_path = root / "TEST_EVALUATION_STARTED.json"
    evaluated_path = root / "TEST_EVALUATED"
    final_summary_path = root / "final_test_summary.json"
    preflight_path = root / "preflight_report.json"
    inventory_path = root / "PREMARKER_CAMPAIGN_INVENTORY.json"
    required = (
        plan_path,
        started_path,
        evaluated_path,
        final_summary_path,
        preflight_path,
        inventory_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing post-test integrity artefacts: {missing}")

    plan = load_json(plan_path)
    started = load_json(started_path)
    final_summary = load_json(final_summary_path)
    inventory = load_json(inventory_path)

    # The frozen protocol uses canonical JSON hashes, not byte-level hashes,
    # for JSON payload identities.
    plan_hash = final_runner.stable_hash(plan)
    if (
        plan.get("protocol") != "confirmatory-test-plan-v2"
        or started.get("campaign_id") != plan.get("campaign_id")
        or started.get("test_plan_sha256") != plan_hash
        or final_summary.get("campaign_id") != plan.get("campaign_id")
        or final_summary.get("test_plan_sha256") != plan_hash
        or final_summary.get("protocol") != "confirmatory-final-summary-v2"
        or inventory.get("campaign_id") != plan.get("campaign_id")
        or inventory.get("test_evidence_detected") is not False
    ):
        raise RuntimeError("Post-test campaign identity or chronology is invalid")

    if (
        final_runner.stable_hash(load_json(preflight_path))
        != plan.get("preflight_report_sha256")
    ):
        raise RuntimeError("Frozen preflight report differs from immutable plan")
    if (
        final_runner.stable_hash(inventory)
        != plan.get("premarker_campaign_inventory_sha256")
    ):
        raise RuntimeError(
            "Frozen pre-marker inventory differs from immutable plan"
        )

    specs = [
        spec
        for dataset_specs in plan["cap_result_specs"].values()
        for spec in dataset_specs.values()
    ] + list(plan["tpe_result_specs"].values())
    if len(specs) != 18:
        raise RuntimeError(f"Unexpected immutable result count: {len(specs)}")
    valid = 0
    for spec in specs:
        result_path = root / "results" / spec["result_filename"]
        if not result_path.is_file():
            raise RuntimeError(f"Missing immutable result: {result_path}")
        valid += int(
            final_runner.validate_result_receipt(
                root,
                result_path,
                spec,
                plan,
                started_path,
            )
        )
    if valid != len(specs):
        raise RuntimeError(
            f"Only {valid}/{len(specs)} immutable receipts are valid"
        )
    return {
        "protocol": "posttest-preflight-wrapper-v1",
        "campaign_id": plan["campaign_id"],
        "test_plan_sha256": plan_hash,
        "valid_receipts": valid,
    }


def main() -> int:
    verification = posttest_preflight(ROOT, DATA_ROOT)
    summarizer.run_preflight = posttest_preflight
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(HERE / "summarize_confirmatory.py"),
            "--data_root",
            str(DATA_ROOT),
            "--output_root",
            str(ROOT),
        ]
        summarizer.main()
    finally:
        sys.argv = original_argv

    status_path = ROOT / "summaries" / "summary_status.json"
    status = load_json(status_path)
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
        key: status.get(key) for key in expected_counts
    }
    if (
        status.get("complete") is not True
        or observed_counts != expected_counts
        or status.get("bound_failures") != 0
        or status.get("final_test_summary_sha256")
        != sha256_file(ROOT / "final_test_summary.json")
        or status.get("test_evaluated_marker_sha256")
        != sha256_file(ROOT / "TEST_EVALUATED")
    ):
        raise RuntimeError(
            "Strict post-test summary audit failed: "
            f"status={status}, expected_counts={expected_counts}"
        )

    audit = {
        **verification,
        "reason": (
            "The frozen pre-test preflight cannot be rerun after official-test "
            "artefacts exist; this wrapper verified the frozen preflight and "
            "pre-marker hashes plus all immutable result receipts before "
            "running the original summarizer in strict mode."
        ),
        "summarizer_sha256": sha256_file(HERE / "summarize_confirmatory.py"),
        "frozen_preflight_sha256": sha256_file(
            HERE / "preflight_confirmatory.py"
        ),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "summary_status_sha256": sha256_file(status_path),
        "completed_unix": time.time(),
    }
    audit_path = ROOT / "POSTTEST_SUMMARY_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "POSTTEST_STRICT_SUMMARY_COMPLETE "
        f"receipts={verification['valid_receipts']} "
        f"status_sha256={audit['summary_status_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
