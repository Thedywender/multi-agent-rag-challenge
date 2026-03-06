"""Handler de ingestão de documentos."""

import uuid

from src.shared.chunking import chunk_text
from src.shared.chroma_client import chroma_add
from src.shared.embeddings import get_embeddings


def handle_ingest(content: str) -> dict:
    """
    Processa o documento: chunking, embeddings e armazenamento no Chroma.

    Args:
        content: Texto do documento.

    Returns:
        Dict com doc_id e chunks_count.
    """
    chunks = chunk_text(content)
    if not chunks:
        return {"doc_id": None, "chunks_count": 0}

    embeddings = get_embeddings(chunks)
    doc_id = str(uuid.uuid4())
    chroma_add(doc_id, chunks, embeddings)

    return {"doc_id": doc_id, "chunks_count": len(chunks)}
