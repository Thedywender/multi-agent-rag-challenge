"""Handler de consulta (RAG)."""

from src.agents.base import retrieve_embedding, build_response
from src.shared.embeddings import get_embeddings
from src.orchestrator.handler import route_question
from src.agents.registry import get_agent


def handle_ask(question: str, k: int = 5) -> dict:
    """
    Responde à pergunta usando RAG: embedding, retrieval e LLM.

    Args:
        question: Pergunta do usuário.
        k: Número de chunks a recuperar.

    Returns:
        Dict com answer e sources.
    """
    q_embedding = get_embeddings([question])[0]
    domain = route_question(question)

    agent = get_agent(domain)
    if agent:
        return agent(question, q_embedding, k=k)

    # fallback geral consulta os dois dominios e responde uma vez
    rh_results = retrieve_embedding(q_embedding, "rh", k=k)
    tecnico_results = retrieve_embedding(q_embedding, "tecnico", k=k)
    return build_response(
        question, (rh_results + tecnico_results)[:k], routed_domain="geral"
    )
