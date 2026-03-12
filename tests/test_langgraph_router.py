from unittest.mock import MagicMock, patch

from src.orchestrator.langgraph_router import (
    answer_node,
    ask_with_langgraph,
    embed_node,
    route_node,
)


def test_embed_node_generates_single_question_embedding():
    state = {
        "question": "Qual a política de férias?",
        "k": 5,
        "embedding": [],
        "domain": "geral",
        "result": {},
    }

    with patch(
        "src.orchestrator.langgraph_router.get_embeddings", return_value=[[0.1, 0.2]]
    ) as get_embeddings:
        out = embed_node(state)

    assert out["embedding"] == [0.1, 0.2]
    get_embeddings.assert_called_once_with(["Qual a política de férias?"])


def test_route_node_sets_domain_from_router():
    state = {
        "question": "Como autenticar no endpoint?",
        "k": 5,
        "embedding": [0.3],
        "domain": "geral",
        "result": {},
    }

    with patch("src.orchestrator.langgraph_router.route_question", return_value="tecnico"):
        out = route_node(state)

    assert out["domain"] == "tecnico"


def test_answer_node_uses_specialist_agent_when_available():
    state = {
        "question": "Pergunta RH",
        "k": 3,
        "embedding": [0.1, 0.9],
        "domain": "rh",
        "result": {},
    }

    agent = MagicMock(return_value={"answer": "ok", "sources": [], "routed_domain": "rh"})

    with patch("src.orchestrator.langgraph_router.get_agent", return_value=agent) as get_agent:
        out = answer_node(state)

    assert out["result"] == {"answer": "ok", "sources": [], "routed_domain": "rh"}
    get_agent.assert_called_once_with("rh")
    agent.assert_called_once_with("Pergunta RH", [0.1, 0.9], 3)


def test_answer_node_fallback_geral_merges_collections_and_caps_k():
    state = {
        "question": "Pergunta ambígua",
        "k": 2,
        "embedding": [0.4, 0.6],
        "domain": "geral",
        "result": {},
    }

    rh_results = [{"document": "rh-1", "metadata": {"domain": "rh"}}]
    tecnico_results = [
        {"document": "tec-1", "metadata": {"domain": "tecnico"}},
        {"document": "tec-2", "metadata": {"domain": "tecnico"}},
    ]

    with patch("src.orchestrator.langgraph_router.get_agent", return_value=None), patch(
        "src.orchestrator.langgraph_router.retrieve_embedding",
        side_effect=[rh_results, tecnico_results],
    ) as retrieve_embedding, patch(
        "src.orchestrator.langgraph_router.build_response",
        return_value={"answer": "geral", "sources": [], "routed_domain": "geral"},
    ) as build_response:
        out = answer_node(state)

    assert out["result"] == {
        "answer": "geral",
        "sources": [],
        "routed_domain": "geral",
    }
    retrieve_embedding.assert_any_call([0.4, 0.6], "rh", k=2)
    retrieve_embedding.assert_any_call([0.4, 0.6], "tecnico", k=2)
    build_response.assert_called_once_with(
        "Pergunta ambígua",
        [
            {"document": "rh-1", "metadata": {"domain": "rh"}},
            {"document": "tec-1", "metadata": {"domain": "tecnico"}},
        ],
        routed_domain="geral",
    )


def test_ask_with_langgraph_invokes_compiled_graph_with_expected_state():
    graph = MagicMock()
    graph.invoke.return_value = {"result": {"answer": "ok", "sources": [], "routed_domain": "rh"}}

    with patch("src.orchestrator.langgraph_router._compiled_graph", return_value=graph):
        out = ask_with_langgraph("qualquer pergunta", k=7)

    assert out == {"answer": "ok", "sources": [], "routed_domain": "rh"}
    graph.invoke.assert_called_once_with(
        {
            "question": "qualquer pergunta",
            "k": 7,
            "embedding": [],
            "domain": "geral",
            "result": {},
        }
    )
