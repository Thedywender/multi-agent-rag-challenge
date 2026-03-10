from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.base import build_response, retrieve_embedding
from src.agents.registry import get_agent
from src.orchestrator.handler import route_question
from src.shared.embeddings import get_embeddings


class AskState(TypedDict):
    question: str
    k: int
    embedding: list[float]
    domain: Literal["rh", "tecnico", "geral"]
    result: dict


def embed_node(state: AskState) -> AskState:
    return {**state, "embedding": get_embeddings([state["question"]])[0]}


def route_node(state: AskState) -> AskState:
    return {**state, "domain": route_question(state["question"])}


def answer_node(state: AskState) -> AskState:
    question = state["question"]
    k = state["k"]
    embedding = state["embedding"]
    domain = state["domain"]

    agent = get_agent(domain)
    if agent:
        return {**state, "result": agent(question, embedding, k)}

    rh_results = retrieve_embedding(embedding, "rh", k=k)
    tecnico_results = retrieve_embedding(embedding, "tecnico", k=k)
    result = build_response(
        question, (rh_results + tecnico_results)[:k], routed_domain="geral"
    )
    return {**state, "result": result}


@lru_cache(maxsize=1)
def _compiled_graph():
    graph = StateGraph(AskState)
    graph.add_node("embed", embed_node)
    graph.add_node("route", route_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("embed")
    graph.add_edge("embed", "route")
    graph.add_edge("route", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def ask_with_langgraph(question: str, k: int = 5) -> dict:
    state = _compiled_graph().invoke(
        {
            "question": question,
            "k": k,
            "embedding": [],
            "domain": "geral",
            "result": {},
        }
    )
    return state["result"]
