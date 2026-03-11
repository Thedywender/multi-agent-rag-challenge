from unittest import patch

from src.agents.rh_agent import answer as rh_answer
from src.agents.tecnico_agent import answer as tecnico_answer
from src.orchestrator.handler import route_question


def test_route_rh_by_kword():
    assert route_question("Qual a política de férias e benefícios?") == "rh"


def test_route_tecnico_by_kword():
    assert route_question("Como autenticar no endpoint com bearer token?") == "tecnico"


def test_route_geral_ambiguous():
    with patch("src.orchestrator.handler._llm_classefy", return_value="geral"):
        assert route_question("beneficios e endpoint na mesma dúvida") == "geral"


def test_rh_agent_uses_only_rh_collection():
    with patch(
        "src.agents.rh_agent.retrieve_embedding", return_value=[]
    ) as retrieve, patch(
        "src.agents.rh_agent.build_response", return_value={"ok": True}
    ) as build:
        out = rh_answer("q", [0.1], k=5)

    assert out == {"ok": True}
    retrieve.assert_called_once_with([0.1], domain="rh", k=5)
    build.assert_called_once()


def test_tecnico_agent_uses_only_tecnico_collection():
    with patch(
        "src.agents.tecnico_agent.retrieve_embedding", return_value=[]
    ) as retrieve, patch(
        "src.agents.tecnico_agent.build_response", return_value={"ok": True}
    ) as build:
        out = tecnico_answer("q", [0.1], k=4)

    assert out == {"ok": True}
    retrieve.assert_called_once_with([0.1], domain="tecnico", k=4)
    build.assert_called_once()
