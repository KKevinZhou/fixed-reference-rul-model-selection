"""Finalize DOI/date metadata before building the immutable Zenodo ZIP."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parent
WORKSPACE = ARTIFACT.parent
SUBMISSION = WORKSPACE / "manuscript"
DOI_PATTERN = re.compile(r"10\.5281/zenodo\.\d+")
DOI_PLACEHOLDER = "[insert Zenodo DOI URL]"
ARTICLE_TITLE = (
    "Fixed-Reference Scoring for Task-Consistent RUL Model Selection "
    "under Tunable Target Capping"
)
SOFTWARE_TITLE = (
    "Fixed-Reference Scoring under Candidate-Dependent RUL Capping: "
    "Reproduction Artifact"
)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_text_batch(payloads: dict[Path, str]) -> None:
    """Validate/stage every text payload before replacing any destination."""
    originals = {path: path.read_bytes() for path in payloads}
    temporaries: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for path, text in payloads.items():
            temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
            temporary.write_text(text, encoding="utf-8")
            temporaries[path] = temporary
        for path, temporary in temporaries.items():
            os.replace(temporary, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            restore = path.with_suffix(path.suffix + f".restore-{os.getpid()}")
            restore.write_bytes(originals[path])
            os.replace(restore, path)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def replace_or_require(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new)
    if new in text:
        return text
    raise RuntimeError(f"{label} contains neither placeholder nor final value")


def render_citation(text: str, doi: str, released: str) -> str:
    existing_dois = set(DOI_PATTERN.findall(text))
    if existing_dois and existing_dois != {doi}:
        raise RuntimeError(
            f"CITATION.cff contains a different Zenodo DOI: {existing_dois}"
        )
    text = re.sub(r"(?m)^date-released:.*\n?", "", text)
    text = re.sub(r"(?m)^doi:.*\n?", "", text)
    marker = (
        "# `date-released` and `doi` are inserted only after a Zenodo DOI "
        "is reserved.\n"
    )
    text = text.replace(marker, "")
    license_line = "license: MIT"
    if text.count(license_line) != 1:
        raise RuntimeError("CITATION.cff license anchor is invalid")
    replacement = (
        f"date-released: {released}\n"
        f'doi: "{doi}"\n'
        f"{license_line}"
    )
    return text.replace(license_line, replacement)


def finalize_citation(path: Path, doi: str, released: str) -> None:
    atomic_text(
        path,
        render_citation(path.read_text(encoding="utf-8"), doi, released),
    )


def render_main(text: str, doi_url: str) -> str:
    replacement = rf"\url{{{doi_url}}} \citep{{jiang2026artifact}}"
    return replace_or_require(
        text, DOI_PLACEHOLDER, replacement, "main.tex"
    )


def finalize_main(path: Path, doi_url: str) -> None:
    atomic_text(
        path,
        render_main(path.read_text(encoding="utf-8"), doi_url),
    )


def render_bibliography(
    text: str, doi: str, doi_url: str, year: int
) -> str:
    existing_dois = set(DOI_PATTERN.findall(text))
    if existing_dois and existing_dois != {doi}:
        raise RuntimeError(
            f"reference.bib contains a different Zenodo DOI: {existing_dois}"
        )
    entry = (
        "\n@misc{jiang2026artifact,\n"
        "  author       = {Jiang, Dejian and Zhou, Zhiheng and Deng, Xiyue},\n"
        f"  title        = {{{SOFTWARE_TITLE}}},\n"
        f"  year         = {{{year}}},\n"
        "  version      = {1.0.0},\n"
        "  publisher    = {Zenodo},\n"
        f"  doi          = {{{doi}}},\n"
        f"  url          = {{{doi_url}}},\n"
        "}\n"
    )
    pattern = re.compile(
        r"\n@misc\{jiang2026artifact,.*?\n\}\n?", flags=re.DOTALL
    )
    if pattern.search(text):
        text = pattern.sub(entry, text)
    else:
        text = text.rstrip() + "\n" + entry
    return text


def finalize_bibliography(path: Path, doi: str, doi_url: str, year: int) -> None:
    atomic_text(
        path,
        render_bibliography(
            path.read_text(encoding="utf-8"), doi, doi_url, year
        ),
    )


def render_cover(text: str, doi_url: str) -> str:
    old = (
        "archived at the Zenodo DOI reported in the manuscript before "
        "submission."
    )
    new = f"archived at {doi_url} before submission."
    return replace_or_require(text, old, new, "cover_letter.md")


def finalize_cover(path: Path, doi_url: str) -> None:
    atomic_text(
        path,
        render_cover(path.read_text(encoding="utf-8"), doi_url),
    )


def prepare_release_texts(
    artifact: Path,
    submission: Path,
    doi: str,
    released: date,
) -> dict[Path, str]:
    """Render every DOI-bound file without mutating the workspace."""
    doi_url = f"https://doi.org/{doi}"
    citation = artifact / "CITATION.cff"
    main = submission / "main.tex"
    bibliography = submission / "reference.bib"
    cover = submission / "cover_letter.md"
    for path in (citation, main, bibliography, cover):
        if not path.is_file():
            raise RuntimeError(f"Required release-metadata file is absent: {path}")
    main_text = main.read_text(encoding="utf-8")
    cover_text = cover.read_text(encoding="utf-8")
    if ARTICLE_TITLE not in re.sub(r"\s+", " ", main_text):
        raise RuntimeError("main.tex does not contain the canonical article title")
    if ARTICLE_TITLE not in re.sub(r"\s+", " ", cover_text):
        raise RuntimeError(
            "cover_letter.md does not contain the canonical article title"
        )
    return {
        citation: render_citation(
            citation.read_text(encoding="utf-8"),
            doi,
            released.isoformat(),
        ),
        main: render_main(main_text, doi_url),
        bibliography: render_bibliography(
            bibliography.read_text(encoding="utf-8"),
            doi,
            doi_url,
            released.year,
        ),
        cover: render_cover(cover_text, doi_url),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", required=True)
    parser.add_argument("--date-released", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT)
    parser.add_argument("--submission-root", type=Path, default=SUBMISSION)
    args = parser.parse_args()

    doi = args.doi.strip().removeprefix("https://doi.org/")
    if DOI_PATTERN.fullmatch(doi) is None:
        raise RuntimeError("DOI must match 10.5281/zenodo.<record-id>")
    try:
        released = date.fromisoformat(args.date_released)
    except ValueError as exc:
        raise RuntimeError("--date-released must be YYYY-MM-DD") from exc
    if released > date.today():
        raise RuntimeError("--date-released cannot be in the future")
    artifact = args.artifact_root.resolve()
    submission = args.submission_root.resolve()
    payloads = prepare_release_texts(
        artifact, submission, doi, released
    )
    atomic_text_batch(payloads)
    doi_url = f"https://doi.org/{doi}"
    print(
        f"FINALIZED release metadata for {doi_url} on "
        f"{released.isoformat()}"
    )


if __name__ == "__main__":
    main()
