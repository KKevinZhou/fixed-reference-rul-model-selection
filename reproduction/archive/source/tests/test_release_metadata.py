from pathlib import Path
from datetime import date

import prepare_release_metadata as metadata


def test_release_metadata_finalization_is_consistent_and_idempotent(tmp_path):
    artifact = tmp_path / "artifact"
    submission = tmp_path / "submission"
    artifact.mkdir()
    submission.mkdir()
    (artifact / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "title: test\n"
        "version: 1.0.0\n"
        "# `date-released` and `doi` are inserted only after a Zenodo DOI "
        "is reserved.\n"
        "license: MIT\n",
        encoding="utf-8",
    )
    (submission / "main.tex").write_text(
        "Artifact: [insert Zenodo DOI URL].\n",
        encoding="utf-8",
    )
    (submission / "reference.bib").write_text(
        "@article{x, title={X}, year={2026}}\n",
        encoding="utf-8",
    )
    (submission / "cover_letter.md").write_text(
        "The versioned reproduction artifact will be\n"
        "archived at the Zenodo DOI reported in the manuscript before "
        "submission.\n",
        encoding="utf-8",
    )
    doi = "10.5281/zenodo.12345678"
    url = f"https://doi.org/{doi}"
    released = "2026-07-25"

    for _ in range(2):
        metadata.finalize_citation(
            artifact / "CITATION.cff", doi, released
        )
        metadata.finalize_main(submission / "main.tex", url)
        metadata.finalize_bibliography(
            submission / "reference.bib", doi, url, 2026
        )
        metadata.finalize_cover(submission / "cover_letter.md", url)

    citation = (artifact / "CITATION.cff").read_text(encoding="utf-8")
    main = (submission / "main.tex").read_text(encoding="utf-8")
    bibliography = (submission / "reference.bib").read_text(
        encoding="utf-8"
    )
    cover = (submission / "cover_letter.md").read_text(encoding="utf-8")
    assert citation.count(f'doi: "{doi}"') == 1
    assert citation.count(f"date-released: {released}") == 1
    assert main.count(url) == 1
    assert main.count(r"\citep{jiang2026artifact}") == 1
    assert bibliography.count("@misc{jiang2026artifact") == 1
    assert bibliography.count(doi) == 2
    assert cover.count(url) == 1


def test_release_metadata_rejects_conflicting_doi(tmp_path):
    citation = Path(tmp_path) / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\n"
        "version: 1.0.0\n"
        'doi: "10.5281/zenodo.111"\n'
        "license: MIT\n",
        encoding="utf-8",
    )
    try:
        metadata.finalize_citation(
            citation, "10.5281/zenodo.222", "2026-07-25"
        )
    except RuntimeError as exc:
        assert "different Zenodo DOI" in str(exc)
    else:
        raise AssertionError("Conflicting DOI was overwritten")


def test_batch_preparation_fails_before_any_file_is_changed(tmp_path):
    artifact = tmp_path / "artifact"
    submission = tmp_path / "submission"
    artifact.mkdir()
    submission.mkdir()
    citation = artifact / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\n"
        "version: 1.0.0\n"
        "# `date-released` and `doi` are inserted only after a Zenodo DOI "
        "is reserved.\n"
        "license: MIT\n",
        encoding="utf-8",
    )
    main = submission / "main.tex"
    main.write_text(
        metadata.ARTICLE_TITLE + "\n[insert Zenodo DOI URL]\n",
        encoding="utf-8",
    )
    (submission / "reference.bib").write_text("", encoding="utf-8")
    cover = submission / "cover_letter.md"
    cover.write_text(
        metadata.ARTICLE_TITLE + "\nmissing release sentence\n",
        encoding="utf-8",
    )
    before = {
        path: path.read_bytes()
        for path in (citation, main, submission / "reference.bib", cover)
    }
    try:
        metadata.prepare_release_texts(
            artifact,
            submission,
            "10.5281/zenodo.12345678",
            date(2026, 7, 25),
        )
    except RuntimeError as exc:
        assert "cover_letter.md" in str(exc)
    else:
        raise AssertionError("Invalid metadata set was accepted")
    assert all(path.read_bytes() == value for path, value in before.items())
