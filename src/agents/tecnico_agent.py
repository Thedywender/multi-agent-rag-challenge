from __future__ import annotations

from src.agents.base import build_response, retrieve_embedding

DOMAIN = "tecnico"


def answer(question: str, question_embedding: list[float], k: int = 5) -> dict:
    """
    Resposta com RAG: retrieval + LLM.

    Args:
        question: Pergunta do usuário.
        k: Número de chunks que devem ser recuperados.

    Returns:
        Dict com answer e sources.
    """
    results = retrieve_embedding(question_embedding, domain=DOMAIN, k=k)
    return build_response(question, results, routed_domain=DOMAIN)
