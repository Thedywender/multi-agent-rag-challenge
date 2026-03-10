from __future__ import annotations

from src.agents.base import build_response, retrieve


def answer(question: str, k: int = 5) -> dict:
    """
    Resposta com RAG: retrieval + LLM.

    Args:
        question: Pergunta do usuário.
        k: Número de chunks que devem ser recuperados.

    Returns:
        Dict com answer e sources.
    """
    domain = "rh"
    results = retrieve(question, domain, k)
    return build_response(question, results, routed_domain=domain)
