"""Handler de consulta (RAG) via LangGraph."""

from src.orchestrator.langgraph_router import ask_with_langgraph


def handle_ask(question: str, k: int = 5) -> dict:
    """
    Responde à pergunta usando o fluxo orquestrado no LangGraph.
    """
    return ask_with_langgraph(question, k=k)
