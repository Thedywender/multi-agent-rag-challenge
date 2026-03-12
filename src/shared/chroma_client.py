"""Cliente Chroma para conexão com o container."""

import logging
import os
from typing import Any, TypedDict

from chromadb.errors import (
    IDAlreadyExistsError,
    DuplicateIDError,
    UniqueConstraintError,
)


import chromadb
from src.shared.validators import (
    validate_content_hash,
    validate_domain,
    validate_doc_id,
    validate_chunks_and_embeddings,
    validate_query_embedding,
    validate_k,
)

logger = logging.getLogger(__name__)


class ExistingDocument(TypedDict):
    doc_id: str
    chunks_count: int
    domain: str


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


def chroma_find_by_content_hash(
    collection_name: str,
    content_hash: str,
) -> ExistingDocument | None:
    validated_domain = validate_domain(collection_name)
    validated_hash = validate_content_hash(content_hash, strict=True)

    client = _get_client()
    resolved_name = _resolve_collection_name(validated_domain)
    collection = client.get_or_create_collection(name=resolved_name)

    results = collection.get(
        where={"content_hash": validated_hash},
        include=["metadatas"],
    )

    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []

    if not ids:
        return None

    if len(metadatas) != len(ids):
        raise ValueError(
            "Inconsistência de metadados no Chroma para content_hash: "
            f"esperado={len(ids)} recebido={len(metadatas)}"
        )

    doc_id_counts: dict[str, int] = {}

    for index, metadata in enumerate(metadatas):
        if not isinstance(metadata, dict):
            raise ValueError(
                "Inconsistência de metadados no Chroma: "
                f"metadata inválido no índice {index} para content_hash={validated_hash[:12]}"
            )

        existing_doc_id = metadata.get("doc_id")
        if not isinstance(existing_doc_id, str) or not existing_doc_id.strip():
            raise ValueError(
                "Inconsistência de metadados no Chroma: "
                f"doc_id ausente/inválido no índice {index} para content_hash={validated_hash[:12]}"
            )

        doc_id_counts[existing_doc_id] = doc_id_counts.get(existing_doc_id, 0) + 1

    doc_id, chunks_count = sorted(
        doc_id_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]

    logger.info(
        "chroma.find_by_content_hash.hit",
        extra={
            "domain": validated_domain,
            "content_hash_prefix": validated_hash[:12],
            "doc_id": doc_id,
            "chunks_count": chunks_count,
        },
    )

    return {
        "doc_id": doc_id,
        "chunks_count": chunks_count,
        "domain": validated_domain,
    }


def chroma_add(
    collection_name: str,
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    content_hash: str,
) -> bool:
    """
    Adiciona chunks e embeddings ao Chroma.

    Args:
        Collection_name: Nome da coleção no Chroma.
        doc_id: ID do documento.
        chunks: Lista de textos dos chunks.
        embeddings: Lista de vetores de embedding.
        hash: string referente ao documento de entrada
    """
    validated_domain = validate_domain(collection_name)
    validated_doc_id = validate_doc_id(doc_id)
    validated_hash = validate_content_hash(content_hash, strict=True)

    validate_chunks_and_embeddings(chunks, embeddings)

    resolved_name = _resolve_collection_name(validated_domain)
    client = _get_client()
    collection = client.get_or_create_collection(name=resolved_name)

    ids = [f"{validated_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": validated_doc_id,
            "domain": validated_domain,
            "content_hash": validated_hash,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(
            "chroma.add.inserted",
            extra={
                "domain": validated_domain,
                "content_hash_prefix": validated_hash[:12],
                "chunks_count": len(chunks),
            },
        )
        return True
    except (IDAlreadyExistsError, DuplicateIDError, UniqueConstraintError):
        logger.info(
            "chroma.add.duplicate",
            extra={
                "domain": validated_domain,
                "content_hash_prefix": validated_hash[:12],
                "chunks_count": len(chunks),
            },
        )
        return False


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
    validated_domain = validate_domain(collection_name)
    validated_query_embedding = validate_query_embedding(query_embedding)
    resolved_name = _resolve_collection_name(validated_domain)
    validated_k = validate_k(k)
    client = _get_client()
    collection = client.get_or_create_collection(name=resolved_name)

    count = collection.count()
    if count == 0:
        return []

    n_results = min(validated_k, count)
    results = collection.query(
        query_embeddings=[validated_query_embedding],
        n_results=n_results,
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

    return [
        {"document": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)
    ]
