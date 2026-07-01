"""tests del hallucination filter"""
from __future__ import annotations

from src.generate.hallucination_filter import (
    contains_memory_claim,
    detect_farewell,
    filter_response,
    has_citations,
)


def test_farewell_hola():
    """saludo simple retorna respuesta corta"""
    assert detect_farewell("Hola") is not None


def test_farewell_with_accent_normalized():
    """"¿Qué tal?" tambien se detecta (el bug real que fixeamos)"""
    assert detect_farewell("¿Qué tal?") is not None


def test_not_farewell_long_message():
    """mensajes largos con "gracias" no son saludo"""
    q = "gracias por tu ayuda con esto pero necesito preguntarte por el manual"
    # este mensaje incluye "gracias" pero tiene mas de 80 chars
    assert detect_farewell(q) is None


def test_memory_claim_detected():
    """detecta "como te mencione antes" """
    text = "como te mencioné antes, el proceso empieza aca"
    assert contains_memory_claim(text)


def test_no_memory_claim_in_normal_text():
    """texto neutro no dispara false positive"""
    assert not contains_memory_claim("el manual dice que apretar el perno a 22 nm")


def test_citations_detected():
    """detecta [1] o (1) como citas"""
    assert has_citations("segun el manual [1], la temperatura es 80C")
    assert has_citations("segun el manual (2), la temperatura es 80C")


def test_no_citations():
    """texto sin marcas no tiene citas"""
    assert not has_citations("segun el manual, la temperatura es 80C")


def test_filter_rejects_memory_claim():
    """filter_response retorna None + razones si hay memory claim"""
    resp, reasons = filter_response("como te mencioné antes, el valor es 5")
    assert resp is None
    assert "memory_claim" in reasons


def test_filter_optionally_requires_citations():
    """si require_citations=True, respuestas sin citas fallan"""
    resp, reasons = filter_response("la temperatura es 80C", require_citations=True)
    assert resp is None
    assert "missing_citations" in reasons


def test_filter_passes_clean_response():
    """respuesta limpia pasa"""
    resp, reasons = filter_response(
        "segun el manual [1], la temperatura es 80C", require_citations=True
    )
    assert resp is not None
    assert reasons == []
