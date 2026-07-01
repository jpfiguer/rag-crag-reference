"""tests de la logica crag de doble juez

usan mocks de LLMJudge para probar las 3 politicas: consenso, disenso, ambos fail
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.generate.crag import (
    CragDecision,
    JudgeResult,
    Verdict,
    evaluate_with_dual_judge,
    parse_judge_response,
)


@dataclass
class FakeJudge:
    provider: str
    fixed_verdict: Verdict
    reason: str = "fixed"

    async def evaluate(self, question, context, answer) -> JudgeResult:
        return JudgeResult(
            verdict=self.fixed_verdict,
            reason=self.reason,
            provider=self.provider,
        )


@pytest.mark.asyncio
async def test_consensus_supported_emits():
    """ambos jueces supported → EMIT"""
    a = FakeJudge("openai", Verdict.SUPPORTED)
    b = FakeJudge("anthropic", Verdict.SUPPORTED)
    outcome = await evaluate_with_dual_judge("q", "c", "a", a, b)
    assert outcome.decision == CragDecision.EMIT
    assert outcome.answer == "a"


@pytest.mark.asyncio
async def test_consensus_unsupported_rejects():
    """ambos jueces unsupported → REJECT"""
    a = FakeJudge("openai", Verdict.UNSUPPORTED)
    b = FakeJudge("anthropic", Verdict.UNSUPPORTED)
    outcome = await evaluate_with_dual_judge("q", "c", "a", a, b)
    assert outcome.decision == CragDecision.REJECT
    assert outcome.answer is None


@pytest.mark.asyncio
async def test_dissent_retries():
    """un juez supported + otro unsupported → RETRY"""
    a = FakeJudge("openai", Verdict.SUPPORTED)
    b = FakeJudge("anthropic", Verdict.UNSUPPORTED)
    outcome = await evaluate_with_dual_judge("q", "c", "a", a, b)
    assert outcome.decision == CragDecision.RETRY
    assert outcome.answer is None


@pytest.mark.asyncio
async def test_judge_error_counts_as_unsupported():
    """si un juez tira excepcion, se cuenta como unsupported (conservador)"""

    class BrokenJudge:
        provider = "broken"

        async def evaluate(self, q, c, a):
            raise RuntimeError("timeout")

    outcome = await evaluate_with_dual_judge(
        "q", "c", "a", BrokenJudge(), FakeJudge("anthropic", Verdict.SUPPORTED)
    )
    # broken (unsupported) + anthropic (supported) → RETRY
    assert outcome.decision == CragDecision.RETRY


def test_parse_pure_json():
    """respuesta json pura se parsea"""
    raw = '{"verdict": "supported", "reason": "todo ok"}'
    r = parse_judge_response(raw)
    assert r is not None
    assert r.verdict == Verdict.SUPPORTED


def test_parse_json_with_preamble():
    """el juez a veces agrega texto antes o despues del json"""
    raw = 'sure, here you go: {"verdict": "unsupported", "reason": "falta"}'
    r = parse_judge_response(raw)
    assert r is not None
    assert r.verdict == Verdict.UNSUPPORTED


def test_parse_invalid_returns_none():
    """respuestas no parseables retornan None"""
    assert parse_judge_response("") is None
    assert parse_judge_response("not json") is None
    assert parse_judge_response('{"verdict": "unknown"}') is None
