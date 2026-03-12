"""Handler de ingestão de documentos."""

import logging
import time
import uuid
from typing import TypedDict

from src.shared.chunking import chunk_text
from src.shared.chroma_client import chroma_add, chroma_find_by_content_hash
from src.shared.embeddings import get_embeddings
from src.shared.validators import (
    validate_chunks_and_embeddings,
    compute_content_hash,
    validate_content,
    validate_domain,
)

logger = logging.getLogger(__name__)


class IngestResult(TypedDict):
    doc_id: str
    chunks_count: int
    domain: str
    already_exists: bool


def handle_ingest(content: str, domain: str) -> IngestResult:
    """
    Processa o documento: chunking, embeddings e armazenamento no Chroma.

    Args:
        content: Texto do documento.

    Returns:
        IngestResult com doc_id, chunks_count, domain e already_exists.
    """
    start_time = time.perf_counter()
    validated_content = validate_content(content)
    validated_domain = validate_domain(domain)

    content_hash = compute_content_hash(validated_content)

    existing = chroma_find_by_content_hash(validated_domain, content_hash)
    if existing is not None:
        result: IngestResult = {
            "doc_id": existing["doc_id"],
            "chunks_count": existing["chunks_count"],
            "domain": validated_domain,
            "already_exists": True,
        }
        logger.info(
            "ingest.completed",
            extra={
                "domain": validated_domain,
                "content_hash_prefix": content_hash[:12],
                "already_exists": True,
                "chunks_count": result["chunks_count"],
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
            },
        )
        return result

    chunks = chunk_text(validated_content)
    embeddings = get_embeddings(chunks)

    validate_chunks_and_embeddings(chunks, embeddings)

    doc_id = str(uuid.uuid4())

    inserted = chroma_add(
        collection_name=validated_domain,
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        content_hash=content_hash,
    )

    if not inserted:
        existing_after = chroma_find_by_content_hash(validated_domain, content_hash)
        if existing_after is None:
            raise RuntimeError(
                "Conflito de inserção detectado, mas o documento não foi encontrado no recheck."
            )
        result = {
            "doc_id": existing_after["doc_id"],
            "chunks_count": existing_after["chunks_count"],
            "domain": validated_domain,
            "already_exists": True,
        }
        logger.info(
            "ingest.completed",
            extra={
                "domain": validated_domain,
                "content_hash_prefix": content_hash[:12],
                "already_exists": True,
                "chunks_count": result["chunks_count"],
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
            },
        )
        return result

    result = {
        "doc_id": doc_id,
        "chunks_count": len(chunks),
        "domain": validated_domain,
        "already_exists": False,
    }
    logger.info(
        "ingest.completed",
        extra={
            "domain": validated_domain,
            "content_hash_prefix": content_hash[:12],
            "already_exists": False,
            "chunks_count": result["chunks_count"],
            "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
        },
    )
    return result
