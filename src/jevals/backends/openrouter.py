"""Decision models on OpenRouter: Jev, Clef, d1, Drex and the rest of its Decisions catalogue.

OpenRouter serves them on the System One wire, so this is the TypeSafe request with an
OpenRouter key and a required model id:

POST https://openrouter.ai/api/v1/systemone
{"model": "cloudflare/clef", "state": ..., "questions": {id: {type, instructions, criteria}}}

Each model has its own price, and the reply's `usage.cost` is what OpenRouter charged, so
that is the cost reported, not a per-token estimate. Not to be confused with the
`openrouter:<model>` spec, which asks a chat LLM to emulate a decision model.
"""

from __future__ import annotations

import os
from typing import Any

from .._types import BackendResponse, Question, Usage
from ._base import Backend, BackendError, http_error
from ._wire import answers_from_typesafe, questions_to_typesafe

OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class OpenRouterDecisionsBackend(Backend):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,  # decision models answer in well under a second; fail fast and retry
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise BackendError("OPENROUTER_API_KEY is not set")
        self.model = model or os.environ.get("OPENROUTER_DECISION_MODEL")
        if not self.model:
            raise BackendError(
                "the decisions backend needs a model: 'decisions:cloudflare/clef', or set "
                "OPENROUTER_DECISION_MODEL (see https://openrouter.ai/models?output_modalities=decisions)"
            )
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.name = f"decisions:{self.model}"
        self.timeout = timeout

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        body = {"model": self.model, "state": state, "questions": questions_to_typesafe(questions)}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = await self._client().post(f"{self.base_url}/v1/systemone", json=body, headers=headers)
        if r.status_code >= 400:
            raise http_error(self.name, r)
        data = r.json()
        usage_raw = data.get("usage") or {}
        cost = usage_raw.get("cost")
        return BackendResponse(
            answers=answers_from_typesafe(data.get("answers") or {}, questions),
            usage=Usage(
                input_tokens=int(usage_raw.get("input_tokens") or 0),
                output_tokens=int(usage_raw.get("output_tokens") or 0),
                cost_usd=float(cost) if cost is not None else None,
                backend=self.name,
            ),
            raw=data,
        )
