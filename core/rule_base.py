"""Centralna baza reguł IF-THEN (jedno źródło prawdy dla silnika i UI)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import default_config as cfg


@dataclass(frozen=True)
class Rule:
    """Reguła: IF error is <temperature_term> AND humidity is <humidity_term> THEN output is <output_term>."""

    rule_id: int
    temperature_term: str
    humidity_term: str
    output_term: str

    @property
    def description(self) -> str:
        return (f"IF Temperature Error is {self.temperature_term} AND Humidity is "
                f"{self.humidity_term} THEN Output is {self.output_term}")


def build_rule_base(table: Optional[dict] = None) -> List[Rule]:
    """Buduje reguły z tabeli (wiersze: temperatura, kolumny: wilgotność); ID od 1."""
    table = cfg.RULE_TABLE if table is None else table
    rules: List[Rule] = []
    rid = 1
    for t_term in cfg.RULE_TEMP_ORDER:
        row = table[t_term]
        if len(row) != len(cfg.RULE_HUM_ORDER):
            raise ValueError(f"Wiersz '{t_term}' ma {len(row)} wpisów, oczekiwano {len(cfg.RULE_HUM_ORDER)}")
        for h_term, out in zip(cfg.RULE_HUM_ORDER, row):
            rules.append(Rule(rid, t_term, h_term, out))
            rid += 1
    return rules


RULES: List[Rule] = build_rule_base()


def rules_to_dataframe(rules: Optional[List[Rule]] = None) -> pd.DataFrame:
    rules = RULES if rules is None else rules
    return pd.DataFrame([{"ID": r.rule_id, "Temperature Error": r.temperature_term,
                          "Humidity": r.humidity_term, "Output": r.output_term,
                          "Rule": r.description} for r in rules])
