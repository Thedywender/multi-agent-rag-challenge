"""Handler de consulta (RAG)."""

from src.shared.chroma_client import chroma_query
from src.shared.embeddings import get_embeddings
from src.orchestrator.handler import route_question
from src.shared.llm import call_llm


def _build_response(question: str, results: list[dict], routed_domain: str) -> dict:
    """
    Monta resposta final, incluindo resposta da LLM e fontes.

    Args:
        question: Pergunta do usuário.
        results: Lista de chunks recuperados do Chroma.
        route_domain: Domínio para roteamento (rh, tecnico ou geral).
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

    if domain == "rh":
        return _build_response(
            question,
            chroma_query("rh", q_embedding, k=k),
            "rh",
        )

    if domain == "tecnico":
        return _build_response(
            question,
            chroma_query("tecnico", q_embedding, k=k),
            "tecnico",
        )

    # fallback geral consulta os dois dominios
    rh_results = chroma_query("rh", q_embedding, k=k)
    tecnico_results = chroma_query("tecnico", q_embedding, k=k)
    return _build_response(question, rh_results + tecnico_results, "geral")
