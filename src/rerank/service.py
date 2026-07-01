"""orquestador de rerank con circuit breaker y fallback en cascada

politica:
1. voyage es el primario (calidad + free tier generoso)
2. cohere es fallback tier 1
3. local (cross-encoder chico en cpu) es el ultimo recurso

el orquestador consulta el circuit breaker antes de llamar a cada
proveedor; si esta abierto lo skippea explicitamente (~0ms) en vez
de pagar el full timeout de la request
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from src.rerank import circuit_breaker as cb
from src.rerank.types import RerankProvider, RerankResult


logger = logging.getLogger(__name__)


class RerankProviderCallable(Protocol):
    """contrato que cumplen todos los proveedores concretos"""

    async def __call__(
        self, query: str, documents: list[str], top_k: int
    ) -> list[RerankResult]: ...


# tiers en orden de preferencia
# se completan con las implementaciones concretas al importar
_TIERS: list[tuple[RerankProvider, RerankProviderCallable]] = []


def register_provider(
    provider: RerankProvider, impl: RerankProviderCallable
) -> None:
    """registra un proveedor en el orden de tiers

    llamar en __init__ del modulo de cada proveedor
    """
    _TIERS.append((provider, impl))


async def rerank(
    query: str,
    documents: list[str],
    top_k: int,
    timeout_s: float = 5.0,
) -> list[RerankResult]:
    """intenta cada tier en orden hasta que uno funcione

    si todos fallan, tira RuntimeError (el caller decide como manejarlo,
    por ejemplo devolver los documentos originales sin reranking)
    """
    for provider, impl in _TIERS:
        if await cb.is_open(provider):
            logger.warning("rerank_skip_open", extra={"provider": provider})
            continue
        try:
            result = await asyncio.wait_for(
                impl(query, documents, top_k), timeout=timeout_s
            )
            await cb.record_success(provider)
            logger.info(
                "rerank_ok",
                extra={"provider": provider, "count": len(result)},
            )
            return result
        except Exception as e:
            await cb.record_failure(provider)
            logger.error(
                "rerank_failed",
                extra={"provider": provider, "error": str(e)},
            )
            continue
    raise RuntimeError("all rerank providers failed or open")
