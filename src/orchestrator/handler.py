"""Orquestrador de roteamento de perguntas por domínio."""

from __future__ import annotations

from typing import Literal

from src.shared.llm import _call_llm_openai

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


def _keyword_score(text: str, keywords: set[str]) -> int:
    """
    Score simples: soma 1 para cada keyword encontrada no texto.
    Isso é melhor que boolean puro e reduz engessamento.
    """
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score


def _llm_classify(question: str) -> Domain:
    """
    Classificação flexível via LLM.
    O LLM deve retornar SOMENTE: rh | tecnico | geral
    """
    prompt = (
        "Classifique a pergunta em apenas uma categoria: rh, tecnico ou geral.\n"
        "Definições:\n"
        "- rh: políticas internas, férias, benefícios, onboarding, home office, regras de RH.\n"
        "- tecnico: APIs, arquitetura, autenticação, integrações, código, infra, endpoints.\n"
        "- geral: não está claro ou mistura os dois.\n\n"
        "Responda SOMENTE com uma dessas palavras: rh, tecnico ou geral.\n\n"
        f"Pergunta: {question}"
    )

    # Se seu call_llm exige context, passe "" como contexto.
    # (mantém compatibilidade com a função existente do projeto)
    raw = _call_llm_openai(prompt, context="")  # ajuste se sua assinatura for diferente
    label = (raw or "").strip().lower()

    if label in ("rh", "tecnico", "geral"):
        return label  # type: ignore[return-value]
    return "geral"


def route_question(question: str) -> Domain:
    """
    Roteamento robusto:
    1) tenta heurística por score
    2) se ambíguo ou fraco, usa LLM para classificar
    3) fallback seguro: geral
    """
    text = question.lower().strip()

    rh_score = _keyword_score(text, RH_KEYWORDS)
    tech_score = _keyword_score(text, TECNICO_KEYWORDS)

    # Casos fortes: diferença clara
    if rh_score and not tech_score:
        return "rh"
    if tech_score and not rh_score:
        return "tecnico"

    # Se um ganhou com folga, roteia pelo vencedor
    if rh_score - tech_score >= 2:
        return "rh"
    if tech_score - rh_score >= 2:
        return "tecnico"

    # Ambíguo ou fraco: pede ajuda ao LLM (mais flexível)
    try:
        return _llm_classify(question)
    except Exception:
        # Nunca quebrar o /ask por falha do LLM do orquestrador
        return "geral"
