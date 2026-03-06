"""Geração de embeddings via OpenAI ou AWS Bedrock."""

import json
import os

from openai import OpenAI

OPENAI_MODEL = "text-embedding-3-small"
BEDROCK_MODEL = "amazon.titan-embed-text-v2:0"
BATCH_SIZE = 100


def _get_provider() -> str:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()
    if provider not in ("openai", "bedrock"):
        raise ValueError(
            "LLM_PROVIDER deve ser 'openai' ou 'bedrock'. "
            f"Valor atual: {provider}"
        )
    return provider


def _get_embeddings_openai(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-"):
        raise ValueError(
            "OPENAI_API_KEY não configurada. Crie o arquivo .env com sua chave da OpenAI."
        )
    client = OpenAI(api_key=api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        batch_embeddings = [
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _get_embeddings_bedrock(texts: list[str]) -> list[list[float]]:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_REGION não configurada. Defina AWS_REGION no .env para usar Bedrock."
        )

    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    all_embeddings: list[list[float]] = []

    for text in texts:
        body = json.dumps({"inputText": text})
        response = client.invoke_model(
            modelId=BEDROCK_MODEL,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        all_embeddings.append(result["embedding"])

    return all_embeddings


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos.

    Usa OpenAI ou AWS Bedrock conforme LLM_PROVIDER (openai|bedrock).

    Args:
        texts: Lista de textos para gerar embeddings.

    Returns:
        Lista de vetores de embedding.
    """
    if not texts:
        return []

    provider = _get_provider()

    if provider == "openai":
        return _get_embeddings_openai(texts)
    return _get_embeddings_bedrock(texts)
