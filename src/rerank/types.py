"""tipos compartidos por el modulo de rerank"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RerankProvider(str, Enum):
    """proveedores de reranking soportados por el orquestador"""

    VOYAGE = "voyage"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass(frozen=True)
class RerankResult:
    """resultado normalizado de cualquier proveedor de rerank

    los proveedores retornan formatos distintos, la capa de servicio
    normaliza a esto para que los consumidores downstream no sepan
    de que proveedor vino
    """

    index: int
    score: float
    document: str
