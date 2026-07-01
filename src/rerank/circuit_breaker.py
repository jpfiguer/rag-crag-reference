"""circuit breaker per-provider para el tier de rerank

trackea fallos consecutivos por proveedor y short-circuitea requests
cuando uno esta degradado. probes periodicos hacen self-heal cuando
la fuente upstream vuelve

por que esto importa: una outage del proveedor primario hace que
cada query pague el full timeout retry antes de fallar al fallback.
con el breaker, despues de N fallos consecutivos skippeamos el proveedor
entero por M segundos, despues probamos con una request unica

threading: el breaker es state a nivel modulo compartido entre asyncio
tasks. un lock protege el dict. no necesita estado cross-process
(redis-backed) porque cada worker se cura solo y no hay invariante
cross-worker en riesgo
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from src.rerank.types import RerankProvider


# abre el breaker despues de N fallos consecutivos en FAIL_WINDOW_S
_FAIL_THRESHOLD = 3
_FAIL_WINDOW_S = 30.0
# cuanto mantiene abierto antes del probe
_RECOVERY_WINDOW_S = 60.0


@dataclass
class _State:
    fail_count: int = 0
    first_fail_ts: float = 0.0
    opened_at: float = 0.0  # 0 means closed


_state: dict[RerankProvider, _State] = {}
_lock = asyncio.Lock()


async def is_open(provider: RerankProvider) -> bool:
    """true si el breaker esta abierto (skip la llamada)"""
    async with _lock:
        st = _state.get(provider)
        if st is None or st.opened_at == 0:
            return False
        # ventana de recovery expiro, half-open: permite un probe
        if time.time() - st.opened_at > _RECOVERY_WINDOW_S:
            return False
        return True


async def record_success(provider: RerankProvider) -> None:
    """resetea el breaker en una llamada exitosa

    llamar despues de cada request exitosa; borra el conteo de fallos
    y cierra el circuito si estaba abierto (post-probe)
    """
    async with _lock:
        st = _state.get(provider)
        if st is None:
            return
        st.fail_count = 0
        st.first_fail_ts = 0.0
        st.opened_at = 0.0


async def record_failure(provider: RerankProvider) -> None:
    """registra un fallo y abre el circuito si supera el threshold

    la ventana se resetea si pasa mas de FAIL_WINDOW_S entre fallos
    (para no acumular fallos viejos que no reflejan el estado actual)
    """
    async with _lock:
        st = _state.setdefault(provider, _State())
        now = time.time()
        # reset ventana si esta stale
        if st.fail_count == 0 or now - st.first_fail_ts > _FAIL_WINDOW_S:
            st.fail_count = 1
            st.first_fail_ts = now
        else:
            st.fail_count += 1
        # abrir el circuito si superamos threshold
        if st.fail_count >= _FAIL_THRESHOLD:
            st.opened_at = now


async def reset(provider: RerankProvider) -> None:
    """reset explicito, usado por tests"""
    async with _lock:
        _state.pop(provider, None)


# thresholds expuestos para tests deterministas
FAIL_THRESHOLD = _FAIL_THRESHOLD
FAIL_WINDOW_S = _FAIL_WINDOW_S
RECOVERY_WINDOW_S = _RECOVERY_WINDOW_S
