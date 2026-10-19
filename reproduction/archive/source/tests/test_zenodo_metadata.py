import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "manuscript"
    / "verify_zenodo_publication.py"
)
SPEC = importlib.util.spec_from_file_location("zenodo_verifier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_legacy_creator_normalization():
    creator = {
        "name": "Jiang, Dejian",
        "orcid": "0009-0000-8939-0903",
        "affiliation": MODULE.AFFILIATION,
    }
    assert MODULE.creator_name(creator) == "Jiang, Dejian"
    assert MODULE.creator_orcid(creator) == "0009-0000-8939-0903"
    assert MODULE.creator_affiliations(creator) == {MODULE.AFFILIATION}


def test_invenio_creator_normalization():
    creator = {
        "person_or_org": {
            "family_name": "Jiang",
            "given_name": "Dejian",
            "identifiers": [
                {
                    "scheme": "orcid",
                    "identifier": "https://orcid.org/0009-0000-8939-0903",
                }
            ],
        },
        "affiliations": [{"name": MODULE.AFFILIATION}],
    }
    assert MODULE.creator_name(creator) == "Jiang, Dejian"
    assert MODULE.creator_orcid(creator) == "0009-0000-8939-0903"
    assert MODULE.creator_affiliations(creator) == {MODULE.AFFILIATION}


def test_metadata_vocabulary_normalization():
    metadata = {
        "rights": [{"id": "mit"}],
        "languages": [{"id": "eng"}],
        "subjects": [{"subject": keyword} for keyword in MODULE.KEYWORDS],
    }
    assert MODULE.metadata_license_ids(metadata) == {"mit"}
    assert MODULE.metadata_languages(metadata) == {"eng"}
    assert MODULE.metadata_keywords(metadata) == MODULE.KEYWORDS


def test_html_description_is_normalized():
    text = MODULE.plain_description(
        "<p>Fixed-reference &amp; C-MAPSS</p><p>reproduction artifact</p>"
    )
    assert text == "Fixed-reference & C-MAPSS reproduction artifact"
