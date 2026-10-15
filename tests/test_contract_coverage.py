from jevals import MockBackend
from jevals._declarative import DeclarativeEval
from jevals.contract import Contract, Rule, check_coverage, rule_id

TEXTS = [
    "Ask for the order number when the customer has not given one.",
    "Never promise a refund before checking that the order is inside the 30-day refund window.",
    "Reply in the customer's language.",
]
ASK, REFUND, LANG = (rule_id(t) for t in TEXTS)
CONTRACT = Contract(prompt_sha256="0" * 64, rules=[Rule(id=rule_id(t), text=t) for t in TEXTS])


def ev(name, questions, **top):
    return DeclarativeEval({"name": name, "questions": questions, **top})


def q(text, covers=None, why=None):
    d = {"type": "noul", "instructions": text}
    if covers is not None:
        d["covers"] = covers
    if why:
        d["why"] = why
    return d


def test_clean_contract():
    evals = [
        ev(
            "support",
            {
                "a": q("Asks for the order number?", [ASK]),
                "b": q("Promises a refund?", [REFUND]),
                "c": q("Same language?", [LANG]),
            },
        )
    ]
    rep = check_coverage(CONTRACT, evals, use_model=False)
    assert rep.uncovered == [] and rep.stale == [] and rep.uncited == [] and not rep.failed(strict=True)


def test_uncovered_rule_fails():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, REFUND])})], use_model=False)
    assert [r.id for r in rep.uncovered] == [LANG] and rep.failed()


def test_stale_citation_fails():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, REFUND, LANG, "r-000000"])})], use_model=False)
    assert [(c.question, c.rule_id) for c in rep.stale] == [("s.a", "r-000000")] and rep.failed()


def test_uncited_fails_only_when_strict_and_observed_counts_as_cited():
    evals = [
        ev("s", {"a": q("?", [ASK, REFUND, LANG]), "b": q("Apologises?", ["observed"], why="tone")}),
        ev("plain", {"x": q("Anything?")}),
    ]
    rep = check_coverage(CONTRACT, evals, use_model=False)
    assert rep.uncited == ["plain.x"]
    assert not rep.failed() and rep.failed(strict=True)


def test_changed_prompt_lists_added_and_removed_rules_and_who_cites_them():
    new_prompt = "- " + TEXTS[0] + "\n- " + TEXTS[2] + "\n- Never share internal ticket ids.\n"
    evals = [ev("s", {"a": q("?", [ASK]), "b": q("?", [REFUND]), "c": q("?", [LANG])})]
    rep = check_coverage(CONTRACT, evals, use_model=False, prompt=new_prompt)
    d = rep.prompt_diff
    assert [r.text for r in d.added] == ["Never share internal ticket ids."]
    assert [r.id for r in d.removed] == [REFUND]
    assert [(c.question, c.rule_id) for c in d.cited_removed] == [("s.b", REFUND)]
    assert rep.failed()


def test_same_prompt_has_no_diff():
    prompt = "\n".join("- " + t for t in TEXTS)
    evals = [ev("s", {"a": q("?", [ASK, REFUND, LANG])})]
    rep = check_coverage(CONTRACT, evals, use_model=False, prompt=prompt)
    assert rep.prompt_diff is not None and not rep.prompt_diff.added and not rep.prompt_diff.removed
    assert not rep.failed()


def test_table_names_each_finding():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, "r-000000"])})], use_model=False)
    t = rep.table()
    assert "uncovered" in t and LANG in t and "stale" in t and "s.a" in t


def test_model_is_not_called_when_use_model_is_false():
    mock = MockBackend()
    check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK])})], backend=mock, use_model=False, prompt=TEXTS[0])
    assert mock.calls == []
