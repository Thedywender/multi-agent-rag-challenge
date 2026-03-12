"""API FastAPI para RAG - ingestão e consulta."""

import logging
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ingest.handler import handle_ingest
from src.query.handler import handle_ask

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="API para ingestão de documentos e consultas com RAG",
    version="1.0.0",
)


class DocumentRequest(BaseModel):
    """Payload para POST /documents."""

    content: str = Field(..., description="Texto do documento")
    domain: Literal["rh", "tecnico"] = Field(..., description="Dominio da coleção")


class AskRequest(BaseModel):
    """Payload para POST /ask."""

    question: str = Field(..., description="Pergunta do usuário")


class DocumentResponse(BaseModel):
    """Resposta de POST /documents."""

    doc_id: str | None
    chunks_count: int
    domain: Literal["rh", "tecnico"]
    already_exists: bool


class SourceItem(BaseModel):
    """Fonte utilizada para responder uma pergunta."""

    document: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    """Resposta de POST /ask."""

    answer: str
    sources: list[SourceItem]
    routed_domain: Literal["rh", "tecnico", "geral"]


class HealthResponse(BaseModel):
    """Resposta de GET /health."""

    status: str


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    field: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "field": field,
            }
        },
    )


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    # ValidationInputError herda de ValueError; detectamos por atributos.
    error_code = getattr(exc, "code", None)
    error_field = getattr(exc, "field", None)

    if error_code is not None:
        logger.warning(
            "validation_error path=%s code=%s field=%s",
            request.url.path,
            error_code,
            error_field,
        )
        return _error_response(
            status_code=400,
            code=str(error_code),
            message=str(exc),
            field=error_field,
        )

    logger.exception("processing_error path=%s", request.url.path)
    return _error_response(
        status_code=503,
        code="service_unavailable",
        message=str(exc),
        field=None,
    )


@app.exception_handler(Exception)
def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected_error path=%s", request.url.path)
    return _error_response(
        status_code=500,
        code="internal_error",
        message="Erro interno inesperado.",
        field=None,
    )


@app.post("/documents", response_model=DocumentResponse)
def post_documents(request: DocumentRequest) -> DocumentResponse:
    """
    Recebe um documento, divide em chunks, gera embeddings e armazena no Chroma.
    """
    result = handle_ingest(request.content, request.domain)
    return DocumentResponse(**result)


@app.post("/ask", response_model=AskResponse)
def post_ask(request: AskRequest) -> AskResponse:
    """
    Responde à pergunta usando RAG: busca contexto no Chroma e gera resposta via LLM.
    """
    result = handle_ask(request.question)
    return AskResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(status="ok")
