"""chunking estructural con page tracking

detecta headings, secciones numeradas y titulos en all-caps para mantener
contenido relacionado junto. cae a sentence-boundary cuando no hay estructura

el extractor de pdf inserta un form-feed marker (\\x0c) despues de cada
pagina; el chunker se acuerda de la pagina por caracter sin cambiar
el texto del chunk. cuando no hay marker (texto plano no-pdf), todo
mapea a pagina 1 y la logica es no-op
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass


_PAGE_MARKER = "\x0c"

_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s+.+", re.MULTILINE),          # markdown headings
    re.compile(r"^\s*\d+(\.\d+){0,3}\s+[A-ZÁÉÍÓÚÑ].+", re.MULTILINE),  # secciones numeradas
    re.compile(r"^\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\-]{4,}$", re.MULTILINE),  # all-caps
]

# tamanio target por chunk (caracteres, no tokens)
_TARGET_SIZE = 800
_MAX_SIZE = 1200


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    heading: str | None
    char_start: int
    char_end: int


def _extract_page_map(raw: str) -> tuple[str, list[int]]:
    """extrae los offsets de cada pagina y retorna el texto sin markers

    el retorno es (texto_limpio, [offset_donde_empieza_pagina_i])
    """
    if _PAGE_MARKER not in raw:
        return raw, [0]

    clean_parts: list[str] = []
    page_starts: list[int] = [0]
    offset = 0
    for piece in raw.split(_PAGE_MARKER):
        clean_parts.append(piece)
        offset += len(piece)
        page_starts.append(offset)
    # el ultimo start es el len final, no una pagina real
    page_starts.pop()
    return "".join(clean_parts), page_starts


def _page_for_offset(page_starts: list[int], offset: int) -> int:
    """dado un offset dentro del texto limpio, retorna el numero de pagina 1-based"""
    idx = bisect.bisect_right(page_starts, offset) - 1
    return max(1, idx + 1)


def _find_boundaries(text: str) -> list[int]:
    """encuentra offsets donde se puede cortar sin partir una unidad semantica

    prioridad: heading > final de parrafo > final de oracion
    """
    boundaries: set[int] = {0, len(text)}
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            boundaries.add(m.start())
    # finales de parrafo
    for m in re.finditer(r"\n\s*\n", text):
        boundaries.add(m.end())
    # finales de oracion
    for m in re.finditer(r"[.!?]\s+", text):
        boundaries.add(m.end())
    return sorted(boundaries)


def _current_heading(text: str, offset: int) -> str | None:
    """busca el ultimo heading antes del offset dado"""
    best_heading: str | None = None
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            if m.start() >= offset:
                break
            best_heading = m.group().strip()
    return best_heading


def chunk_text(raw: str) -> list[Chunk]:
    """chunkea el texto respetando estructura y page tracking"""
    text, page_starts = _extract_page_map(raw)
    boundaries = _find_boundaries(text)

    chunks: list[Chunk] = []
    i = 0
    n = len(text)
    while i < n:
        target_end = min(i + _TARGET_SIZE, n)
        # buscar el boundary mas cercano a target_end sin pasarse de _MAX_SIZE
        j = bisect.bisect_right(boundaries, i + _MAX_SIZE) - 1
        # candidato: el boundary mas grande que sea <= target_end
        candidates = [b for b in boundaries if i < b <= i + _MAX_SIZE]
        chunk_end = target_end
        for c in candidates:
            if c >= target_end:
                chunk_end = c
                break
        if chunk_end == i:
            chunk_end = min(i + _MAX_SIZE, n)
        chunk_text = text[i:chunk_end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page=_page_for_offset(page_starts, i),
                    heading=_current_heading(text, i),
                    char_start=i,
                    char_end=chunk_end,
                )
            )
        i = chunk_end
    return chunks
