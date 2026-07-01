"""tests del circuit breaker

usan asyncio y monkey-patchean el clock donde hace falta para no
depender de sleeps reales
"""
from __future__ import annotations

import asyncio

import pytest

from src.rerank import circuit_breaker as cb
from src.rerank.types import RerankProvider


@pytest.mark.asyncio
async def test_starts_closed():
    """en el estado inicial no hay proveedor abierto"""
    await cb.reset(RerankProvider.VOYAGE)
    assert not await cb.is_open(RerankProvider.VOYAGE)


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    """despues de FAIL_THRESHOLD fallos consecutivos el breaker abre"""
    await cb.reset(RerankProvider.VOYAGE)
    for _ in range(cb.FAIL_THRESHOLD):
        await cb.record_failure(RerankProvider.VOYAGE)
    assert await cb.is_open(RerankProvider.VOYAGE)


@pytest.mark.asyncio
async def test_success_resets_the_state():
    """un exito borra la cuenta de fallos y cierra el circuito"""
    await cb.reset(RerankProvider.VOYAGE)
    for _ in range(cb.FAIL_THRESHOLD - 1):
        await cb.record_failure(RerankProvider.VOYAGE)
    await cb.record_success(RerankProvider.VOYAGE)
    assert not await cb.is_open(RerankProvider.VOYAGE)


@pytest.mark.asyncio
async def test_probe_after_recovery_window(monkeypatch):
    """despues de RECOVERY_WINDOW_S el circuito permite un probe (is_open false)"""
    await cb.reset(RerankProvider.VOYAGE)
    for _ in range(cb.FAIL_THRESHOLD):
        await cb.record_failure(RerankProvider.VOYAGE)
    assert await cb.is_open(RerankProvider.VOYAGE)

    # simulamos que paso el tiempo con monkeypatch del clock
    real_time = __import__("time").time
    offset = cb.RECOVERY_WINDOW_S + 1
    monkeypatch.setattr(cb.time, "time", lambda: real_time() + offset)

    assert not await cb.is_open(RerankProvider.VOYAGE)


@pytest.mark.asyncio
async def test_independent_state_per_provider():
    """abrir voyage no afecta cohere"""
    await cb.reset(RerankProvider.VOYAGE)
    await cb.reset(RerankProvider.COHERE)
    for _ in range(cb.FAIL_THRESHOLD):
        await cb.record_failure(RerankProvider.VOYAGE)
    assert await cb.is_open(RerankProvider.VOYAGE)
    assert not await cb.is_open(RerankProvider.COHERE)


@pytest.mark.asyncio
async def test_stale_failures_reset_window():
    """fallos separados por mas de FAIL_WINDOW_S no se acumulan"""
    await cb.reset(RerankProvider.VOYAGE)
    await cb.record_failure(RerankProvider.VOYAGE)

    # simulamos que paso mas de FAIL_WINDOW_S entre fallos
    real_time = __import__("time").time
    offset = cb.FAIL_WINDOW_S + 1

    import time as _t
    original = _t.time
    _t.time = lambda: original() + offset

    try:
        await cb.record_failure(RerankProvider.VOYAGE)
        # solo hay 1 fallo efectivo, no deberia estar abierto
        assert not await cb.is_open(RerankProvider.VOYAGE)
    finally:
        _t.time = original
