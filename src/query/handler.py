"""Handler de consulta (RAG)."""

from src.shared.chroma_client import chroma_query
from src.shared.embeddings import get_embeddings
from src.shared.llm import call_llm


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
    results = chroma_query(q_embedding, k=k)

    if not results:
        return {
            "answer": "Não há documentos indexados. Envie documentos via POST /documents primeiro.",
            "sources": [],
        }

    context = "\n\n".join(r["document"] for r in results)
    answer = call_llm(question, context)

    return {
        "answer": answer,
        "sources": [{"document": r["document"], "metadata": r["metadata"]} for r in results],
    }
