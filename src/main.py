"""API FastAPI para RAG - ingestão e consulta."""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ingest.handler import handle_ingest
from src.query.handler import handle_ask

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


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
    )


@app.post("/documents", response_model=DocumentResponse)
def post_documents(request: DocumentRequest) -> DocumentResponse:
    """
    Recebe um documento, divide em chunks, gera embeddings e armazena no Chroma.
    """
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=400, detail="O campo 'content' não pode estar vazio"
        )

    result = handle_ingest(request.content, request.domain)
    return DocumentResponse(**result)


@app.post("/ask", response_model=AskResponse)
def post_ask(request: AskRequest) -> AskResponse:
    """
    Responde à pergunta usando RAG: busca contexto no Chroma e gera resposta via LLM.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400, detail="O campo 'question' não pode estar vazio"
        )

    result = handle_ask(request.question)
    return AskResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(status="ok")
