from __future__ import annotations

from pathlib import Path

import pytest

from rag_hpo.ontology import acquire_ontology, build_entries

OBO = """format-version: 1.2
ontology: hp

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0000118
name: Phenotypic abnormality
is_a: HP:0000001 ! All

[Term]
id: HP:0001945
name: Fever
def: "An elevation of body temperature." []
synonym: "Pyrexia" EXACT []
is_a: HP:0000118 ! Phenotypic abnormality
"""


def test_build_entries_includes_labels_synonyms_and_addons(tmp_path: Path) -> None:
    obo = tmp_path / "hp.obo"
    obo.write_text(OBO, encoding="utf-8")
    addons = tmp_path / "addons.csv"
    addons.write_text(
        "HP_ID,info\nHP:0001945,High temperature\nHP:9999999,Unknown\n",
        encoding="utf-8",
    )
    entries = build_entries(obo, addons_path=addons, limit=None)
    fever = [entry for entry in entries if entry.hp_id == "HP:0001945"]
    assert {entry.phrase for entry in fever} == {
        "Fever",
        "Pyrexia",
        "High temperature",
    }
    assert {entry.source for entry in fever} == {
        "hpo-label",
        "hpo-synonym",
        "hpo-addon",
    }


def test_limit_is_deterministic(tmp_path: Path) -> None:
    obo = tmp_path / "hp.obo"
    obo.write_text(OBO, encoding="utf-8")
    entries = build_entries(obo, addons_path=None, limit=1)
    assert len(entries) == 1
    assert entries[0].hp_id == "HP:0000001"


def test_acquire_local_cached_and_offline(tmp_path: Path) -> None:
    supplied = tmp_path / "supplied.obo"
    supplied.write_text(OBO, encoding="utf-8")
    path, source = acquire_ontology(
        output_dir=tmp_path,
        obo_file=supplied,
        obo_url="https://example.test/hp.obo",
        refresh=False,
        offline=True,
    )
    assert path == supplied
    assert source == str(supplied.resolve())

    cached = tmp_path / "hp.obo"
    cached.write_text(OBO, encoding="utf-8")
    path, source = acquire_ontology(
        output_dir=tmp_path,
        obo_file=None,
        obo_url="https://example.test/hp.obo",
        refresh=False,
        offline=True,
    )
    assert path == cached
    assert source == "https://example.test/hp.obo"

    cached.unlink()
    with pytest.raises(FileNotFoundError, match="offline"):
        acquire_ontology(
            output_dir=tmp_path,
            obo_file=None,
            obo_url="https://example.test/hp.obo",
            refresh=False,
            offline=True,
        )


def test_acquire_rejects_missing_supplied_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        acquire_ontology(
            output_dir=tmp_path,
            obo_file=tmp_path / "missing.obo",
            obo_url="https://example.test/hp.obo",
            refresh=False,
            offline=False,
        )
