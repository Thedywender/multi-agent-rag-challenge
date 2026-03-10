from __future__ import annotations

from typing import Callable

from src.agents.rh_agent import answer as answer_rh
from src.agents.tecnico_agent import answer as answer_tecnico

AgentHandler = Callable[[str, list[float], int], dict]

AGENT_REGISTRY: dict[str, AgentHandler] = {
    "rh": answer_rh,
    "tecnico": answer_tecnico,
}


def get_agent(domain: str) -> AgentHandler | None:
    return AGENT_REGISTRY.get(domain)
