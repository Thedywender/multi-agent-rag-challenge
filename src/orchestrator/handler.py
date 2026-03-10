"""Orquestrador de roteamento de perguntas por domínio."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from src.shared.llm import (
    call_llm_context,
)

Domain = Literal["rh", "tecnico", "geral"]

RH_KEYWORDS = {
    "rh",
    "recursos humanos",
    "ferias",
    "beneficio",
    "beneficios",
    "onboarding",
    "gestor",
    "colaborador",
    "colaboradores",
    "home office",
    "plr",
    "dress code",
    "jornada",
    "folha",
    "holerite",
    "admissao",
    "demissao",
}

TECNICO_KEYWORDS = {
    "api",
    "endpoint",
    "integracao",
    "oauth",
    "token",
    "arquitetura",
    "microservico",
    "kubernetes",
    "docker",
    "postgres",
    "redis",
    "rabbitmq",
    "webhook",
    "payload",
    "autenticacao",
    "sdk",
    "request",
    "response",
    "latencia",
    "infra",
}

MIN_SCORE_TO_DECIDE = 1
MIN_MARGIN_TO_DECIDE = 2


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)


def _keyword_score(text: str, keywords: set[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def _llm_classify(question: str) -> Domain:
    prompt = (
        "Classifique a pergunta em apenas uma categoria: rh, tecnico ou geral.\n"
        "Definições:\n"
        "- rh: políticas internas, férias, benefícios, onboarding, home office, regras de RH.\n"
        "- tecnico: APIs, arquitetura, autenticação, integrações, código, infra, endpoints.\n"
        "- geral: não está claro ou mistura os dois.\n\n"
        "Responda SOMENTE com uma dessas palavras: rh, tecnico ou geral.\n\n"
        f"Pergunta: {question}"
    )

    raw = call_llm_context(prompt)
    label = (raw or "").strip().lower()

    if label in ("rh", "tecnico", "geral"):
        return label  # type: ignore[return-value]

    # fallback tolerante
    if "tecnico" in label:
        return "tecnico"
    if "rh" in label:
        return "rh"
    return "geral"


def route_question(question: str) -> Domain:
    text = _normalize(question)
    if not text:
        return "geral"

    rh_score = _keyword_score(text, RH_KEYWORDS)
    tech_score = _keyword_score(text, TECNICO_KEYWORDS)

    # Se houver um lado com kw e o outro não
    if rh_score >= MIN_SCORE_TO_DECIDE and tech_score == 0:
        return "rh"
    if tech_score >= MIN_SCORE_TO_DECIDE and rh_score == 0:
        return "tecnico"

    # Confirmando se não tem ambiguidade entre os dois
    if rh_score - tech_score >= MIN_MARGIN_TO_DECIDE:
        return "rh"
    if tech_score - rh_score >= MIN_MARGIN_TO_DECIDE:
        return "tecnico"

    # Se ambiguo manda pro LLM
    try:
        return _llm_classify(question)
    except Exception:
        return "geral"
