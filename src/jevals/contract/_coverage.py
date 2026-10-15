"""Which prompt rules have a question, and which questions still match the prompt."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .._declarative import DeclarativeEval
from .._types import Noul, Usage
from ..backends import Backend, resolve
from ..backends._base import run_coro_sync
from ._rules import Contract, Rule, aextract_rules

OBSERVED = "observed"


class Citation(BaseModel):
    question: str  # "<eval>.<qid>"
    rule_id: str


class Doubt(BaseModel):
    question: str
    rule_id: str
    p: float


class PromptDiff(BaseModel):
    added: list[Rule]
    removed: list[Rule]
    cited_removed: list[Citation]


class CoverageReport(BaseModel):
    uncovered: list[Rule] = []
    stale: list[Citation] = []
    prompt_diff: PromptDiff | None = None
    uncited: list[str] = []
    doubtful: list[Doubt] = []
    usage: Usage = Field(default_factory=Usage)

    def failed(self, strict: bool = False) -> bool:
        changed = self.prompt_diff is not None and bool(self.prompt_diff.added or self.prompt_diff.removed)
        hard = bool(self.uncovered or self.stale or changed)
        return hard or (strict and bool(self.uncited or self.doubtful))

    def table(self) -> str:
        rows: list[str] = []
        for r in self.uncovered:
            rows.append(f"uncovered   {r.id}  {r.text}")
        for c in self.stale:
            rows.append(f"stale       {c.question} cites {c.rule_id}, which the contract does not have")
        if self.prompt_diff:
            for r in self.prompt_diff.added:
                rows.append(f"new rule    {r.id}  {r.text}  (not in the contract)")
            for r in self.prompt_diff.removed:
                rows.append(f"gone rule   {r.id}  {r.text}  (no longer in the prompt)")
            for c in self.prompt_diff.cited_removed:
                rows.append(f"              cited by {c.question}")
        for qid in self.uncited:
            rows.append(f"uncited     {qid}  (no covers)")
        for d in self.doubtful:
            rows.append(f"doubtful    {d.question} -> {d.rule_id}  p={d.p:.2f}")
        head = "contract coverage: clean" if not rows else f"contract coverage: {len(rows)} finding(s)"
        tail = f"\n{self.usage}" if self.usage.requests else ""
        return "\n".join([head, *rows]) + tail


def _citations(evals: Sequence[DeclarativeEval]) -> tuple[list[Citation], list[str]]:
    cites: list[Citation] = []
    uncited: list[str] = []
    for ev in evals:
        for qid, ids in getattr(ev, "covers", {}).items():
            name = f"{ev.name}.{qid}"
            if not ids:
                uncited.append(name)
            cites.extend(Citation(question=name, rule_id=i) for i in ids if i != OBSERVED)
    return cites, uncited


async def acheck_coverage(
    contract: Contract,
    evals: Sequence[DeclarativeEval],
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    prompt: str | None = None,
) -> CoverageReport:
    cites, uncited = _citations(evals)
    known = contract.ids()
    cited_ids = {c.rule_id for c in cites}
    rep = CoverageReport(
        uncovered=[r for r in contract.rules if r.id not in cited_ids],
        stale=[c for c in cites if c.rule_id not in known],
        uncited=uncited,
    )
    if prompt is not None:
        ex = await aextract_rules(prompt, backend=backend, use_model=use_model)
        now = ex.contract.ids()
        removed = [r for r in contract.rules if r.id not in now]
        rep.prompt_diff = PromptDiff(
            added=[r for r in ex.contract.rules if r.id not in known],
            removed=removed,
            cited_removed=[c for c in cites if c.rule_id in {r.id for r in removed}],
        )
        rep.usage = rep.usage + ex.usage
    pairs = [c for c in cites if c.rule_id in known]
    if use_model and pairs:
        state = {
            "rules": {rid: contract.get(rid).text for rid in sorted({c.rule_id for c in pairs})},
            "questions": {q: _instructions(evals, q) for q in sorted({c.question for c in pairs})},
        }
        questions = {
            f"p{i}": Noul(
                f"Does questions[{c.question!r}] test whether the agent follows rules[{c.rule_id!r}]?"
            )
            for i, c in enumerate(pairs)
        }
        resp = await resolve(backend).run(state, questions)
        rep.usage = rep.usage + resp.usage
        for i, c in enumerate(pairs):
            p = resp.answers[f"p{i}"].probability
            if p < 0.5:
                rep.doubtful.append(Doubt(question=c.question, rule_id=c.rule_id, p=p))
    return rep


def _instructions(evals: Sequence[DeclarativeEval], question: str) -> str:
    name, _, qid = question.partition(".")
    for ev in evals:
        if ev.name == name and qid in ev._questions:
            ins = ev._questions[qid].instructions
            return ins if isinstance(ins, str) else str(ins)
    return ""


def check_coverage(
    contract: Contract,
    evals: Sequence[DeclarativeEval],
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    prompt: str | None = None,
) -> CoverageReport:
    return run_coro_sync(
        acheck_coverage(contract, evals, backend=backend, use_model=use_model, prompt=prompt)
    )
