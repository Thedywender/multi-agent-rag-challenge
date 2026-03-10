"""Funções base compartilhadas entre agentes especialistas."""

from __future__ import annotations

from src.shared.chroma_client import chroma_query
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


def retrieve_embedding(
    question_embedding: list[float], domain: str, k: int
) -> list[dict]:
    """
    Executa retrieval embedding em um domínio específico.
    """

    return chroma_query(domain, question_embedding, k=k)
