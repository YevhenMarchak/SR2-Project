from config import default_config as cfg
from core.rule_base import RULES, build_rule_base, rules_to_dataframe


def test_rule_count_and_complete_grid():
    assert len(RULES) == 20
    combos = {(r.temperature_term, r.humidity_term) for r in RULES}
    assert len(combos) == 20
    assert combos == {(t, h) for t in cfg.TEMPERATURE_TERMS for h in cfg.HUMIDITY_TERMS}


def test_ids_unique_and_consecutive():
    assert [r.rule_id for r in RULES] == list(range(1, 21))


def test_consequents_exist():
    assert all(r.output_term in cfg.OUTPUT_TERMS for r in RULES)


def test_no_duplicate_rules():
    assert len({(r.temperature_term, r.humidity_term, r.output_term) for r in RULES}) == 20


def test_direction_consistency():
    heat = {"Strong Heating", "Moderate Heating", "Gentle Heating"}
    cool = {"Strong Cooling", "Moderate Cooling", "Gentle Cooling"}
    for r in RULES:
        if r.temperature_term in ("Very Cold", "Cold"):
            assert r.output_term in heat
        if r.temperature_term in ("Warm", "Very Warm"):
            assert r.output_term in cool


def test_humidity_alone_cannot_cause_strong_action_at_comfort():
    for r in RULES:
        if r.temperature_term == "Comfort":
            assert r.output_term in ("Neutral", "Gentle Cooling")


def test_dataframe_matches_rules_and_description():
    df = rules_to_dataframe()
    assert len(df) == 20 and list(df["ID"]) == [r.rule_id for r in RULES]
    assert "IF Temperature Error is Very Cold" in RULES[0].description


def test_build_from_custom_table_validates_row_length():
    import pytest
    bad = {k: v[:3] for k, v in cfg.RULE_TABLE.items()}
    with pytest.raises(ValueError):
        build_rule_base(bad)
