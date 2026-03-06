"""API FastAPI para RAG - ingestão e consulta."""

from fastapi import FastAPI, HTTPException
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


class AskRequest(BaseModel):
    """Payload para POST /ask."""

    question: str = Field(..., description="Pergunta do usuário")


@app.exception_handler(ValueError)
def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
    )


@app.post("/documents")
def post_documents(request: DocumentRequest) -> dict:
    """
    Recebe um documento, divide em chunks, gera embeddings e armazena no Chroma.
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="O campo 'content' não pode estar vazio")

    result = handle_ingest(request.content)
    return result


@app.post("/ask")
def post_ask(request: AskRequest) -> dict:
    """
    Responde à pergunta usando RAG: busca contexto no Chroma e gera resposta via LLM.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="O campo 'question' não pode estar vazio")

    result = handle_ask(request.question)
    return result


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}
