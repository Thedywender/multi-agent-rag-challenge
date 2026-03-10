"""Funções base compartilhadas entre agentes especialistas."""

from __future__ import annotations

from src.shared.chroma_client import chroma_query
from src.shared.embeddings import get_embeddings
from src.shared.llm import call_llm


def build_response(question: str, results: list[dict], routed_domain: str) -> dict:
    """
    Monta resposta final (LLM + fontes).
    Mantém formato esperado por /ask.
    """
    if not results:
        return {
            "answer": "Não encontrei documentos relevantes para essa pergunta.",
            "sources": [],
            "routed_domain": routed_domain,
        }

    context = "\n\n".join(r["document"] for r in results)
    answer = call_llm(question, context)

    return {
        "answer": answer,
        "sources": [
            {"document": r["document"], "metadata": r["metadata"]} for r in results
        ],
        "routed_domain": routed_domain,
    }


def retrieve(question: str, domain: str, k: int) -> list[dict]:
    """
    Executa retrieval em um domínio específico.
    """
    q_embedding = get_embeddings([question])[0]
    return chroma_query(domain, q_embedding, k=k)
