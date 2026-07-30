import json

import pytest

from rag_hpo.registry import HPOTermRegistry


def test_dynamic_hpo_acronym_extraction() -> None:
    """Verify acronyms are parsed dynamically from raw HPO metadata."""
    mock_hpo_data = {
        "HP:0001963": {
            "name": "Postural orthostatic tachycardia syndrome",
            "is_obsolete": False,
            "synonyms": ["POTS", "Postural tachycardia syndrome"],
        },
        "HP:0000083": {
            "name": "Renal insufficiency",
            "is_obsolete": False,
            "synonyms": ["AKI"],
        },
    }
    registry = HPOTermRegistry(raw_ontology_terms=mock_hpo_data)

    expanded = registry.expand_acronyms("Patient evaluated for POTS and AKI.")
    assert "POTS (Postural orthostatic tachycardia syndrome)" in expanded
    assert "AKI (Renal insufficiency)" in expanded


def test_external_dictionary_ingestion(tmp_path: pytest.TempPathFactory) -> None:
    """Verify loading from external UMLS/custom JSON files."""
    dict_file = tmp_path / "medical_acronyms.json"
    dict_file.write_text(
        json.dumps(
            {
                "ESR": "Erythrocyte sedimentation rate",
                "CRP": "C-reactive protein",
            }
        ),
        encoding="utf-8",
    )

    registry = HPOTermRegistry(acronym_dictionary_path=str(dict_file))
    expanded = registry.expand_acronyms("Elevated ESR and CRP levels.")

    assert "ESR (Erythrocyte sedimentation rate)" in expanded
    assert "CRP (C-reactive protein)" in expanded


def test_obsolete_resolution_chain_and_cycles() -> None:
    """Verify multi-step resolution and circular dependency protection."""
    mock_data = {
        "HP:0000001": {"is_obsolete": True, "replaced_by": "HP:0000002"},
        "HP:0000002": {"is_obsolete": True, "consider": ["HP:0000003"]},
        "HP:0000003": {"name": "Active Node", "is_obsolete": False},
        # Circular test case
        "HP:0009991": {"is_obsolete": True, "replaced_by": "HP:0009992"},
        "HP:0009992": {"is_obsolete": True, "replaced_by": "HP:0009991"},
    }
    registry = HPOTermRegistry(raw_ontology_terms=mock_data)

    assert registry.resolve_term_id("HP:0000001") == "HP:0000003"
    # Should safely terminate loop without throwing recursion error
    assert registry.resolve_term_id("HP:0009991") in ["HP:0009991", "HP:0009992"]
