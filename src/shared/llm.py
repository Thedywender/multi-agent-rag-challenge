"""Chamada ao LLM via OpenAI ou AWS Bedrock."""

import json
import os

from openai import OpenAI

OPENAI_MODEL = "gpt-4o-mini"
BEDROCK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

SYSTEM_PROMPT = """Você é um assistente que responde perguntas com base exclusivamente no contexto fornecido.
- Se o contexto contiver informação que responde à pergunta, responda citando ou parafraseando o que está no contexto.
- Não avalie se o contexto é "suficiente", "justificado" ou "objetivo" — use o que está escrito.
- Diga que não encontrou a informação apenas quando o contexto não mencionar nada relevante para a pergunta."""


def _get_provider() -> str:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()
    if provider not in ("openai", "bedrock"):
        raise ValueError(
            "LLM_PROVIDER deve ser 'openai' ou 'bedrock'. " f"Valor atual: {provider}"
        )
    return provider


def call_llm_contexto_openai(prompt: str) -> str:
    """
    Chama o LLM para tarefas sem contexto RAG (ex.: classificação).
    Não use isso para responder perguntas com fontes.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-"):
        raise ValueError("OPENAI_API_KEY não configurada.")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Você é um classificador. Responda apenas com a label solicitada.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # classificação determinística
    )
    return resp.choices[0].message.content.strip() or ""


def _call_llm_openai(question: str, context: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-"):
        raise ValueError(
            "OPENAI_API_KEY não configurada. Crie o arquivo .env com sua chave da OpenAI."
        )
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Contexto:\n\n{context}\n\nPergunta: {question}",
            },
        ],
    )

    return response.choices[0].message.content.strip() or ""


def _call_llm_bedrock(question: str, context: str) -> str:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_REGION não configurada. Defina AWS_REGION no .env para usar Bedrock."
        )

    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": f"Contexto:\n\n{context}\n\nPergunta: {question}",
                }
            ],
        }
    )

    response = client.invoke_model(
        modelId=BEDROCK_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    content = result.get("content", [])
    if content and isinstance(content[0], dict):
        return content[0].get("text", "") or ""
    return ""


def call_llm(question: str, context: str) -> str:
    """
    Chama o LLM com a pergunta e o contexto retornado pela busca.

    Usa OpenAI ou AWS Bedrock conforme LLM_PROVIDER (openai|bedrock).

    Args:
        question: Pergunta do usuário.
        context: Contexto relevante dos documentos.

    Returns:
        Resposta gerada pelo LLM.
    """
    if not context.strip():
        return "Não encontrei documentos relevantes para responder à sua pergunta."

    provider = _get_provider()

    if provider == "openai":
        return _call_llm_openai(question, context)
    return _call_llm_bedrock(question, context)
