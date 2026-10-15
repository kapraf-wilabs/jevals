"""An agent's system prompt as a list of rules with stable ids.

The id is a hash of the rule's normalised text, so a rule keeps its id when the prompt is
reordered and gets a new one when it is reworded. That is what lets `check` say which
tests went stale.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from .._text import split_sentences

RULE_ID = re.compile(r"^r-[0-9a-f]{6}$")


def normalize_rule(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip().lower())
    return s.rstrip(" .!?;:")


def rule_id(text: str) -> str:
    return "r-" + hashlib.sha256(normalize_rule(text).encode()).hexdigest()[:6]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


class Rule(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not RULE_ID.match(v):
            raise ValueError(f"rule id {v!r} must look like r-3f2a9c")
        return v


class Contract(BaseModel):
    prompt_sha256: str
    source: str | None = None
    rules: list[Rule]

    def ids(self) -> set[str]:
        return {r.id for r in self.rules}

    def get(self, rid: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rid), None)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True, width=110)

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())
        return p


def load_contract(path: str | Path) -> Contract:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with prompt_sha256 and rules")
    return Contract.model_validate(data)


def split_rules(prompt: str) -> list[str]:
    """Candidate rules: sentences and list items, headings dropped, duplicates removed."""
    out: list[str] = []
    seen: set[str] = set()
    for line in split_sentences(prompt):
        if line.lstrip().startswith("#"):
            continue
        key = normalize_rule(line)
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return out


def prompt_from_trace(path: str | Path, index: int = 0) -> str:
    from .._runner import load_jsonl
    from .._sample import as_sample

    rows = load_jsonl(path)
    if not 0 <= index < len(rows):
        raise ValueError(f"{path} has {len(rows)} traces; no trace at index {index}")
    prompt = as_sample(rows[index]).system_prompt
    if not prompt:
        raise ValueError(f"{path} trace {index} has no system prompt")
    return prompt
