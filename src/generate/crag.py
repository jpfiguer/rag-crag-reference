"""CRAG (Corrective RAG) con doble juez

despues de generar la respuesta, dos jueces LLM de familias distintas
scorean si la respuesta esta sustentada por el contexto recuperado

por que dos: un solo juez tiene su propio bias del proveedor. dos de
familias distintas (openai + anthropic) bajan el sesgo y dan senal de
duda cuando disienten (mejor rechazar que emitir con baja confianza)

politica de decision:
  - ambos ok  → emitir respuesta
  - ambos fail → rechazar y reintentar (con mas contexto o reformulando)
  - disenso    → rechazar y reintentar (tratamos disenso como duda)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """veredicto de un juez sobre una respuesta"""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class JudgeResult:
    """resultado de un juez"""

    verdict: Verdict
    reason: str
    provider: str


class LLMJudge(Protocol):
    """contrato que cumple cualquier juez"""

    provider: str

    async def evaluate(
        self, question: str, context: str, answer: str
    ) -> JudgeResult: ...


class CragDecision(str, Enum):
    """decision final del orquestador crag"""

    EMIT = "emit"
    RETRY = "retry"
    REJECT = "reject"


@dataclass(frozen=True)
class CragOutcome:
    """salida del pipeline crag con detalles de la decision"""

    decision: CragDecision
    judges: list[JudgeResult]
    answer: str | None


_SYSTEM_PROMPT = (
    "sos un evaluador estricto. te doy una PREGUNTA, un CONTEXTO recuperado "
    "y una RESPUESTA generada. tu unica tarea es decidir si la RESPUESTA esta "
    "SUSTENTADA por el CONTEXTO. una respuesta esta sustentada solo si cada "
    "afirmacion factual puede derivarse literalmente del CONTEXTO. "
    "responde SOLO json: {\"verdict\": \"supported\" | \"unsupported\", "
    "\"reason\": \"<oracion corta explicando\"}"
)


def build_prompt(question: str, context: str, answer: str) -> str:
    """arma el user prompt del juez con formato uniforme"""
    return (
        f"PREGUNTA:\n{question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"RESPUESTA:\n{answer}"
    )


async def evaluate_with_dual_judge(
    question: str,
    context: str,
    answer: str,
    judge_a: LLMJudge,
    judge_b: LLMJudge,
) -> CragOutcome:
    """corre ambos jueces en paralelo y decide

    los jueces corren en paralelo con asyncio.gather; si uno tira
    excepcion se contabiliza como unsupported para conservadurismo
    """
    import asyncio

    async def _safe(j: LLMJudge) -> JudgeResult:
        try:
            return await j.evaluate(question, context, answer)
        except Exception as e:
            logger.error(
                "judge_error",
                extra={"provider": j.provider, "error": str(e)},
            )
            return JudgeResult(
                verdict=Verdict.UNSUPPORTED,
                reason=f"error del juez: {e}",
                provider=j.provider,
            )

    a, b = await asyncio.gather(_safe(judge_a), _safe(judge_b))

    if a.verdict == Verdict.SUPPORTED and b.verdict == Verdict.SUPPORTED:
        decision = CragDecision.EMIT
    elif a.verdict == Verdict.UNSUPPORTED and b.verdict == Verdict.UNSUPPORTED:
        decision = CragDecision.REJECT
    else:
        # disenso: tratamos como duda, reintentamos
        decision = CragDecision.RETRY

    logger.info(
        "crag_decision",
        extra={
            "decision": decision.value,
            "judge_a": {"provider": a.provider, "verdict": a.verdict.value},
            "judge_b": {"provider": b.provider, "verdict": b.verdict.value},
        },
    )

    return CragOutcome(
        decision=decision,
        judges=[a, b],
        answer=answer if decision == CragDecision.EMIT else None,
    )


def parse_judge_response(raw: str) -> JudgeResult | None:
    """parsea la respuesta json del juez de forma robusta

    acepta tanto json puro como json rodeado de texto explicativo
    (los llms a veces agregan preambulo aunque pidas solo json)
    """
    import re

    raw = (raw or "").strip()
    if not raw:
        return None
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict):
        return None
    verdict_raw = obj.get("verdict")
    if verdict_raw not in ("supported", "unsupported"):
        return None
    return JudgeResult(
        verdict=Verdict(verdict_raw),
        reason=str(obj.get("reason", "")),
        provider="parsed",
    )
