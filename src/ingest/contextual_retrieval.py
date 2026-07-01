"""contextual retrieval estilo anthropic

antes de embedear cada chunk, un llm chico genera 50-100 tokens de
anclaje situacional (donde vive el chunk en el documento). eso resuelve
el problema del "chunk lacks situational context":
un fragmento como "apretar el perno M8 a 22 Nm" sin decir de que
documento, seccion o paso proviene

impacto reportado (paper anthropic):
  - 49% menos retrievals fallidos solo
  - 67% menos con reranking encima

costo bajo con prompt caching:
  - el documento completo va en el bloque cacheado (TTL 5 min)
  - solo cambia chunk + question entre calls
  - re-ingesta de 1500 chunks en ~15-30 min por ~$1 con anthropic haiku
"""
from __future__ import annotations

import logging
from typing import Protocol


logger = logging.getLogger(__name__)


# trim del documento en el system block cacheado
# el documento full puede ser 100k+ tokens; recortar a ~30k mantiene
# el cache footprint razonable y la latencia predecible
_DOC_MAX_CHARS = 30_000
# hard cap del contexto retornado ~ 400 chars = 50 tokens
_CONTEXT_MAX_CHARS = 400


_SYSTEM_PROMPT = (
    "sos un experto tecnico que genera ANCLAJE SITUACIONAL para fragmentos de "
    "documentacion tecnica.\n\n"
    "recibiras:\n"
    "- DOCUMENTO completo (en el bloque cacheado)\n"
    "- FRAGMENTO especifico extraido del documento\n\n"
    "tu tarea: redactar UN parrafo de 1-3 oraciones (max 50 palabras) que ubique "
    "el fragmento dentro del documento, mencionando:\n"
    "- que seccion o tema trata\n"
    "- que documento (nombre corto o tipo)\n"
    "- que informacion clave sostiene el fragmento"
)


class LLMWithCache(Protocol):
    """contrato del llm con soporte para prompt caching

    implementaciones concretas: AnthropicHaikuClient, OpenAIMiniClient
    """

    provider: str

    async def generate_with_cached_system(
        self,
        cached_system: str,
        user_message: str,
        max_tokens: int,
    ) -> str: ...


async def generate_situational_context(
    document_text: str,
    chunk_text: str,
    llm: LLMWithCache,
) -> str:
    """genera un anclaje situacional para el chunk usando el llm

    el system block contiene el documento completo (cached) y el system prompt;
    solo el user message cambia por chunk. eso permite reusar el cache anthropic
    de 5 min TTL entre llamadas consecutivas para el mismo documento
    """
    doc_trimmed = document_text[:_DOC_MAX_CHARS]
    cached_system = f"{_SYSTEM_PROMPT}\n\n=== DOCUMENTO ===\n{doc_trimmed}"

    user_msg = f"=== FRAGMENTO ===\n{chunk_text}\n\nGenera el anclaje situacional"

    try:
        raw = await llm.generate_with_cached_system(
            cached_system=cached_system,
            user_message=user_msg,
            max_tokens=120,
        )
    except Exception as e:
        logger.warning(
            "contextual_retrieval_fallback",
            extra={"provider": llm.provider, "error": str(e)},
        )
        # fallback conservador: sin contexto en vez de romper la ingesta
        return ""

    return (raw or "").strip()[:_CONTEXT_MAX_CHARS]


def build_context_prefix(
    *, situational_context: str, metadata: dict[str, str] | None = None
) -> str:
    """combina anclaje + metadata en un prefijo para prepender al chunk antes de embedear

    orden: metadata (mas grosero) primero, contexto situacional despues
    ambos se separan del chunk con doble newline para que el modelo
    de embedding los perciba como bloques
    """
    parts: list[str] = []
    if metadata:
        parts.append(
            " · ".join(f"{k}: {v}" for k, v in metadata.items() if v)
        )
    if situational_context:
        parts.append(situational_context)
    return "\n\n".join(parts)
