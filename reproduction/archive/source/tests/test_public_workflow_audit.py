import json
from pathlib import Path

from audit_public_workflow import validate_receipt


def test_pinned_public_workflow_receipt_matches_audit_constants():
    artifact = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (artifact / "public_workflow_audit.json").read_text(
            encoding="utf-8"
        )
    )
    validate_receipt(payload)
