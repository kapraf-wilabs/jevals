import json

import yaml

from jevals.cli import main

PROMPT = "# Agent\n\n- Ask for the order number.\n- Reply in the customer's language.\n"


def test_rules_then_check_round_trip(tmp_path, capsys):
    (tmp_path / "p.md").write_text(PROMPT)
    out = tmp_path / "contract.yaml"
    args = ["--backend", "mock", "contract", "rules", str(tmp_path / "p.md"), "-o", str(out), "--no-model"]
    assert main(args) == 0
    ids = [r["id"] for r in yaml.safe_load(out.read_text())["rules"]]
    assert len(ids) == 2
    evals = tmp_path / "evals"
    evals.mkdir()
    one = {"name": "e", "questions": {"a": {"type": "noul", "instructions": "Asks?", "covers": [ids[0]]}}}
    (evals / "e.yaml").write_text(yaml.safe_dump(one))
    assert main(["--backend", "mock", "contract", "check", str(out), str(evals), "--no-model"]) == 1
    assert ids[1] in capsys.readouterr().out
    both = {"name": "e", "questions": {"a": {"type": "noul", "instructions": "Asks?", "covers": ids}}}
    (evals / "e.yaml").write_text(yaml.safe_dump(both))
    assert main(["--backend", "mock", "contract", "check", str(out), str(evals), "--no-model", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["uncovered"] == []


def test_rules_prints_yaml_to_stdout_and_reads_a_trace(tmp_path, capsys):
    t = tmp_path / "t.jsonl"
    msgs = [{"role": "system", "content": PROMPT}, {"role": "user", "content": "x"}]
    t.write_text(json.dumps({"messages": msgs}) + "\n")
    assert main(["--backend", "mock", "contract", "rules", "--from-trace", str(t), "--no-model"]) == 0
    assert "Ask for the order number." in yaml.safe_load(capsys.readouterr().out)["rules"][0]["text"]


def test_usage_errors_exit_2(tmp_path, capsys):
    assert main(["--backend", "mock", "contract", "rules", "--no-model"]) == 2
    assert (
        main(["--backend", "mock", "contract", "check", str(tmp_path / "missing.yaml"), str(tmp_path)]) == 2
    )
    assert "error" in capsys.readouterr().err
