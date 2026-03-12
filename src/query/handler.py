"""Handler de consulta (RAG) via LangGraph."""

from src.orchestrator.langgraph_router import ask_with_langgraph
from src.shared.validators import validate_question


def handle_ask(question: str, k: int = 5) -> dict:
    """
    Responde à pergunta usando o fluxo orquestrado no LangGraph.
    """
    validated_question = validate_question(question)
    return ask_with_langgraph(validated_question, k=k)
