import importlib.util
import json
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[1]
SUBMISSION = ARTIFACT.parent / "manuscript"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zenodo_release_record_is_hash_bound_and_unapproved(tmp_path):
    module = load_module(
        "prepare_zenodo_release_approval_record_test",
        SUBMISSION / "prepare_zenodo_release_approval_record.py",
    )
    template = tmp_path / "template.json"
    template.write_text(
        (
            SUBMISSION / "ZENODO_RELEASE_APPROVALS.template.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release = tmp_path / "release.zip"
    metadata = tmp_path / "metadata.md"
    citation = tmp_path / "CITATION.cff"
    release.write_bytes(b"release")
    metadata.write_text("metadata", encoding="utf-8")
    citation.write_text("citation", encoding="utf-8")

    draft_metadata = tmp_path / "ZENODO_DRAFT_METADATA.json"
    draft_metadata.write_text('{"record_id": "123"}', encoding="utf-8")

    payload = module.build_pending_payload(
        template, release, metadata, citation, draft_metadata
    )
    assert payload["confirmed_utc"] == ""
    assert payload["approved_scope"] == {
        "release_zip_sha256": module.sha256_file(release),
        "zenodo_metadata_sha256": module.sha256_file(metadata),
        "citation_cff_sha256": module.sha256_file(citation),
        "zenodo_draft_metadata_sha256": module.sha256_file(draft_metadata),
    }
    assert payload["publication_authorized"] is False
    assert payload["authorized_by"] == ""
    assert payload["authorized_utc"] == ""
    assert all(
        row["release_archive_approved"] is False
        and row["zenodo_metadata_approved"] is False
        for row in payload["authors"]
    )
    json.dumps(payload)


def test_zenodo_draft_snapshot_binds_record_metadata_and_archive(tmp_path):
    module = load_module(
        "prepare_zenodo_draft_metadata_record_test",
        SUBMISSION / "prepare_zenodo_draft_metadata_record.py",
    )
    template = tmp_path / "template.json"
    template.write_text(
        (
            SUBMISSION / "ZENODO_DRAFT_METADATA.template.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release = tmp_path / "candidate-dependent-target-capping-artifact-v1.0.0.zip"
    release.write_bytes(b"exact release")
    payload = module.build_pending_payload(
        template,
        release,
        "12345678",
        "https://doi.org/10.5281/zenodo.12345678",
        "2026-07-25",
    )
    assert payload["record_id"] == "12345678"
    assert payload["reserved_doi"] == "10.5281/zenodo.12345678"
    assert payload["files"] == [
        {
            "name": release.name,
            "bytes": release.stat().st_size,
            "sha256": module.sha256_file(release),
        }
    ]
    assert payload["verified_against_zenodo_preview_by"] == ""
    assert payload["verified_utc"] == ""


def test_submission_audit_validates_verified_zenodo_draft_snapshot(tmp_path):
    audit = load_module(
        "audit_submission_zenodo_draft_test",
        SUBMISSION / "audit_submission.py",
    )
    release = tmp_path / "candidate-dependent-target-capping-artifact-v1.0.0.zip"
    release.write_bytes(b"exact release")
    snapshot = tmp_path / "ZENODO_DRAFT_METADATA.json"
    payload = json.loads(
        (
            SUBMISSION / "ZENODO_DRAFT_METADATA.template.json"
        ).read_text(encoding="utf-8")
    )
    payload.update(
        {
            "record_id": "12345678",
            "reserved_doi": "10.5281/zenodo.12345678",
            "publication_date": "2026-07-25",
            "files": [
                {
                    "name": release.name,
                    "bytes": release.stat().st_size,
                    "sha256": audit.sha256_file(release),
                }
            ],
            "verified_against_zenodo_preview_by": "Dejian Jiang",
            "verified_utc": "2026-07-25T18:00:00Z",
        }
    )
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    audit.FINAL_RELEASE = release
    audit.ZENODO_DRAFT_METADATA = snapshot
    errors = []
    observed = audit.audit_zenodo_draft_metadata(errors)
    assert errors == []
    assert observed == payload

    payload["title"] = "tampered"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    errors = []
    assert audit.audit_zenodo_draft_metadata(errors) is None
    assert any("artifact title mismatch" in error for error in errors)
