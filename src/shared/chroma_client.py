"""Cliente Chroma para conexão com o container."""

import os
from typing import Any

import chromadb

COLLECTION_MAP = {
    "rh": "rh_docs",  # Chroma impõe regra de nome de (coleção >= 3) chars
    "tecnico": "tecnico_docs",  # Para padronizar os nomes
}


def _get_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=host, port=port)


def _resolve_collection_name(domain: str) -> str:
    normalized = (domain or "").strip().lower()
    return COLLECTION_MAP.get(normalized, normalized)


def chroma_add(
    collection_name: str, doc_id: str, chunks: list[str], embeddings: list[list[float]]
) -> None:
    """
    Adiciona chunks e embeddings ao Chroma.

    Args:
        Collection_name: Nome da coleção no Chroma.
        doc_id: ID do documento.
        chunks: Lista de textos dos chunks.
        embeddings: Lista de vetores de embedding.
    """
    resolved_name = _resolve_collection_name(collection_name)
    client = _get_client()
    collection = client.get_or_create_collection(name=resolved_name)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "domain": collection_name} for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def chroma_query(
    collection_name: str, query_embedding: list[float], k: int = 5
) -> list[dict[str, Any]]:
    """
    Busca os k chunks mais similares à query.

    Args:
        collection_name: Nome da coleção no Chroma.
        query_embedding: Vetor de embedding da pergunta.
        k: Número de resultados a retornar.

    Returns:
        Lista de dicts com 'document' e 'metadata'.
    """
    resolved_name = _resolve_collection_name(collection_name)
    client = _get_client()
    collection = client.get_or_create_collection(name=resolved_name)

    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

    return [
        {"document": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)
    ]
