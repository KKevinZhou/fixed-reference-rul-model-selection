import importlib.util
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[1]
SUBMISSION = ARTIFACT.parent / "manuscript"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ai_assisted_graphical_abstract_is_not_a_submission_asset():
    package = load_module(
        "ress_build_submission_package",
        SUBMISSION / "build_submission_package.py",
    )
    audit = load_module(
        "ress_audit_submission",
        SUBMISSION / "audit_submission.py",
    )
    asset_writer = load_module(
        "ress_asset_manifest",
        SUBMISSION / "write_submission_asset_manifest.py",
    )
    assert "Graphical Abstract" not in package.SEPARATE_UPLOADS
    assert "graphical_abstract.tex" not in audit.ASSET_FILES
    assert "graphical_abstract.pdf" not in audit.ASSET_FILES
    assert "graphical_abstract.tex" not in asset_writer.ASSET_FILES
    assert "graphical_abstract.pdf" not in asset_writer.ASSET_FILES
    assert "graphical_abstract.tex" not in (
        ARTIFACT / "wait_and_finalize_confirmatory.py"
    ).read_text(encoding="utf-8")
    upload_map = (SUBMISSION / "UPLOAD_MAP.md").read_text(
        encoding="utf-8"
    )
    assert "| Graphical Abstract |" not in upload_map
    assert "No graphical abstract is submitted" in upload_map
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    assert "no generative image model was used" in manuscript


def test_zenodo_doi_extraction_normalizes_cff_and_url_forms():
    audit = load_module(
        "ress_audit_submission_doi",
        SUBMISSION / "audit_submission.py",
    )
    expected = "https://doi.org/10.5281/zenodo.12345678"
    assert (
        audit.extract_zenodo_doi_url(expected, cff=False)
        == expected
    )
    assert (
        audit.extract_zenodo_doi_url(
            'doi: "10.5281/zenodo.12345678"\n', cff=True
        )
        == expected
    )
    assert (
        audit.extract_zenodo_doi_url(
            'doi: "10.5281/zenodo.12345678"\n', cff=False
        )
        is None
    )


def test_submission_source_archive_covers_all_editable_tex_documents():
    package = load_module(
        "ress_build_submission_package_sources",
        SUBMISSION / "build_submission_package.py",
    )
    expected = {
        "main.tex",
        "supplementary_material.tex",
        "generated_supplementary_tables.tex",
        "generated_carrier_transfer_table.tex",
        "title_page.tex",
        "highlights.tex",
    }
    assert expected.issubset(set(package.SOURCE_FILES))
    source = (SUBMISSION / "build_submission_package.py").read_text(
        encoding="utf-8"
    )
    for stem in (
        "main",
        "supplementary_material",
        "title_page",
        "highlights",
    ):
        assert f'"{stem}"' in source
