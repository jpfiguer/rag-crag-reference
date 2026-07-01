"""fastapi app minimo que expone el pipeline rag como servicio http

endpoints:
  POST /query    ejecuta el pipeline rag completo
  GET  /health   liveness probe
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.generate.hallucination_filter import detect_farewell, filter_response
from src.logging_setup import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    require_citations: bool = True
    top_k: int = 8


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = []
    decision: str


app = FastAPI(title="rag-crag-reference", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """liveness probe sin dependencias externas"""
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest) -> QueryResponse:
    """ejecuta el pipeline rag y devuelve la respuesta con citas

    politica:
    1. si es saludo/despedida, respuesta fija sin llamar al llm
    2. embed + retrieve + rerank (con circuit breaker)
    3. generar respuesta
    4. crag con doble juez
    5. hallucination filter

    en esta version reference, los pasos 2-4 estan stubbeados con TODOs
    para que la app arranque y sirva de punto de partida al lector
    """
    # bypass rapido para saludos: mejor ux y $0
    farewell = detect_farewell(req.question)
    if farewell:
        return QueryResponse(answer=farewell, decision="farewell_bypass")

    # aca iria: embed(question) → qdrant.search → rerank → context builder →
    # generate → dual_judge → filter_response
    # ver los modulos correspondientes para los patrones concretos
    generated = "TODO: integrar los proveedores reales de llm, embedding, qdrant"

    filtered, reasons = filter_response(
        generated, require_citations=req.require_citations
    )
    if filtered is None:
        raise HTTPException(status_code=422, detail={"reasons": reasons})

    return QueryResponse(answer=filtered, decision="emit")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.app_host, port=settings.app_port)
