from pathlib import Path

import pytest

from jevals.contract import (
    Contract,
    Rule,
    load_contract,
    normalize_rule,
    prompt_from_trace,
    rule_id,
    split_rules,
)

PROMPT = """# Acme support agent

You are the support agent for Acme, an online store for kitchen equipment.

## Rules

- Ask for the order number when the customer has not given one.
- Look the order up with `lookup_order` before you answer any question about it.
- Never promise a refund before checking that the order is inside the 30-day refund window.
- Escalate to a human when the customer asks for more than $200 back.
- Reply in the customer's language.
"""


def test_rule_id_is_stable_under_case_space_and_trailing_punctuation():
    a = rule_id("Reply in the customer's language.")
    assert a == rule_id("  reply   in the CUSTOMER'S language  ")
    assert a == "r-9756ad"
    assert a != rule_id("Reply in English.")
    assert normalize_rule(" Ask  for it! ") == "ask for it"


def test_split_drops_headings_keeps_code_spans_and_dedupes():
    lines = split_rules(PROMPT + "\n- Reply in the customer's language.\n")
    assert lines[0].startswith("You are the support agent")
    assert not any(line.lstrip().startswith("#") for line in lines)
    assert "Look the order up with `lookup_order` before you answer any question about it." in lines
    assert lines.count("Reply in the customer's language.") == 1
    assert len(lines) == 6


def test_split_numbered_list():
    lines = split_rules("Rules:\n1. Be brief in every reply.\n2) Never share internal ticket ids.")
    assert "Be brief in every reply." in lines and "Never share internal ticket ids." in lines


def test_contract_round_trips_through_yaml(tmp_path: Path):
    c = Contract(
        prompt_sha256="ab" * 32,
        source="prompt.md",
        rules=[Rule(id=rule_id("Be brief."), text="Be brief.")],
    )
    path = c.dump(tmp_path / "contract.yaml")
    back = load_contract(path)
    assert back == c and back.ids() == {rule_id("Be brief.")}
    assert back.get(rule_id("Be brief.")).text == "Be brief." and back.get("r-000000") is None


def test_load_contract_rejects_a_bad_id(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("prompt_sha256: x\nrules:\n  - id: rule-1\n    text: Be brief.\n")
    with pytest.raises(ValueError, match="rule-1"):
        load_contract(p)


def test_prompt_from_trace(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}\n'
    )
    assert prompt_from_trace(p) == "Be brief."
    q = tmp_path / "none.jsonl"
    q.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    with pytest.raises(ValueError, match="system prompt"):
        prompt_from_trace(q)
