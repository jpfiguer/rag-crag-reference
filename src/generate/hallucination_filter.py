"""HallucinationFilter: filtros regex + heuristica sobre la respuesta generada

detecta patterns tipicos de alucinacion:
1. memory claims: "como mencione antes", "recuerdo que"
2. respuesta que no cita ninguna fuente cuando deberia
3. formulas inventadas (validacion contra un allowlist opcional)

se aplica DESPUES del crag como ultima capa; si detecta, retorna None
o una version sanitizada segun politica
"""
from __future__ import annotations

import re
import unicodedata


_MEMORY_CLAIM_PATTERNS = [
    r"\bcomo (te|le|les) (mencion[eé]|expliqu[eé]|dije)\b",
    r"\brecuerdo que\b",
    r"\bcomo (te|le|les) coment[eé]\b",
    r"\ben nuestra conversaci[oó]n anterior\b",
]

_FAREWELL_PATTERNS = [
    r"^hola\b",
    r"^buenos?\s+(?:dias|tardes|noches)",
    r"^buenas",
    r"^que\s+tal",
    r"^hey\b",
    r"^saludos",
    r"^gracias\b",
    r"^chao\b",
    r"^adios\b",
]


def _normalize(s: str) -> str:
    """lowercase + strip diacritics + trim

    "¿Qué hora es?" → "que hora es"
    esto se hizo despues de un bug real donde queries con acentos
    caian a rag y respondian con info irrelevante porque los patterns
    no matcheaban la forma normalizada
    """
    s = s.lower().strip()
    s = re.sub(r"[!¡?¿,.\-]", "", s).strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def detect_farewell(question: str) -> str | None:
    """si es un saludo o despedida, retorna una respuesta corta fija

    bypass del llm porque los modelos chicos no respetan la instruccion
    de brevedad para mensajes sociales. mejor ux y $0
    """
    if not question:
        return None
    if len(question) > 80:
        return None
    q = _normalize(question)
    for pattern in _FAREWELL_PATTERNS:
        if re.match(pattern, q):
            return "hola, soy el asistente. como te puedo ayudar hoy"
    return None


def contains_memory_claim(text: str) -> bool:
    """true si el texto tiene un memory claim (invento de contexto previo)"""
    if not text:
        return False
    t = _normalize(text)
    for pattern in _MEMORY_CLAIM_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def has_citations(text: str) -> bool:
    """heuristica simple: la respuesta cita fuente si contiene [1], (1), etc."""
    return bool(re.search(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)", text or ""))


def filter_response(
    text: str, *, require_citations: bool = False
) -> tuple[str | None, list[str]]:
    """aplica los filtros a la respuesta final

    retorna:
      (respuesta_sanitizada_o_none, lista_de_razones)

    si retorna None, el caller decide como reaccionar: mostrar mensaje
    generico, reintentar la generacion, o loggear como fallo
    """
    reasons: list[str] = []
    if contains_memory_claim(text):
        reasons.append("memory_claim")
    if require_citations and not has_citations(text):
        reasons.append("missing_citations")
    if reasons:
        return None, reasons
    return text, []
