from __future__ import annotations

from src.agents.base import build_response, retrieve


def answer(question: str, k: int = 5) -> dict:
    domain = "tecnico"
    results = retrieve(question, domain=domain, k=k)
    return build_response(question, results, routed_domain=domain)
