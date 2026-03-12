from unittest.mock import patch

import pytest

from src.ingest.handler import handle_ingest


def test_handle_ingest_returns_existing_document_without_reembedding():
    existing = {"doc_id": "doc-existing", "chunks_count": 2, "domain": "rh"}

    with patch("src.ingest.handler.validate_content", return_value="conteudo"), patch(
        "src.ingest.handler.validate_domain", return_value="rh"
    ), patch("src.ingest.handler.compute_content_hash", return_value="a" * 64), patch(
        "src.ingest.handler.chroma_find_by_content_hash", return_value=existing
    ) as find_existing, patch(
        "src.ingest.handler.chunk_text"
    ) as chunk_text, patch(
        "src.ingest.handler.get_embeddings"
    ) as get_embeddings:
        out = handle_ingest("conteudo", "rh")

    assert out == {
        "doc_id": "doc-existing",
        "chunks_count": 2,
        "domain": "rh",
        "already_exists": True,
    }
    find_existing.assert_called_once_with("rh", "a" * 64)
    chunk_text.assert_not_called()
    get_embeddings.assert_not_called()


def test_handle_ingest_inserts_new_document():
    fixed_uuid = "11111111-1111-1111-1111-111111111111"

    with patch("src.ingest.handler.validate_content", return_value="conteudo"), patch(
        "src.ingest.handler.validate_domain", return_value="tecnico"
    ), patch("src.ingest.handler.compute_content_hash", return_value="b" * 64), patch(
        "src.ingest.handler.chroma_find_by_content_hash", return_value=None
    ) as find_existing, patch(
        "src.ingest.handler.chunk_text", return_value=["c1", "c2"]
    ) as chunk_text, patch(
        "src.ingest.handler.get_embeddings", return_value=[[0.1], [0.2]]
    ) as get_embeddings, patch(
        "src.ingest.handler.validate_chunks_and_embeddings"
    ) as validate_pair, patch(
        "src.ingest.handler.uuid.uuid4", return_value=fixed_uuid
    ), patch(
        "src.ingest.handler.chroma_add", return_value=True
    ) as chroma_add:
        out = handle_ingest("conteudo", "tecnico")

    assert out == {
        "doc_id": fixed_uuid,
        "chunks_count": 2,
        "domain": "tecnico",
        "already_exists": False,
    }
    find_existing.assert_called_once_with("tecnico", "b" * 64)
    chunk_text.assert_called_once_with("conteudo")
    get_embeddings.assert_called_once_with(["c1", "c2"])
    validate_pair.assert_called_once_with(["c1", "c2"], [[0.1], [0.2]])
    chroma_add.assert_called_once_with(
        collection_name="tecnico",
        doc_id=fixed_uuid,
        chunks=["c1", "c2"],
        embeddings=[[0.1], [0.2]],
        content_hash="b" * 64,
    )


def test_handle_ingest_duplicate_insert_returns_existing_after_recheck():
    with patch("src.ingest.handler.validate_content", return_value="conteudo"), patch(
        "src.ingest.handler.validate_domain", return_value="rh"
    ), patch("src.ingest.handler.compute_content_hash", return_value="c" * 64), patch(
        "src.ingest.handler.chroma_find_by_content_hash",
        side_effect=[None, {"doc_id": "doc-race", "chunks_count": 1, "domain": "rh"}],
    ) as find_existing, patch(
        "src.ingest.handler.chunk_text", return_value=["c1"]
    ), patch(
        "src.ingest.handler.get_embeddings", return_value=[[0.5]]
    ), patch(
        "src.ingest.handler.validate_chunks_and_embeddings"
    ), patch(
        "src.ingest.handler.uuid.uuid4", return_value="22222222-2222-2222-2222-222222222222"
    ), patch(
        "src.ingest.handler.chroma_add", return_value=False
    ):
        out = handle_ingest("conteudo", "rh")

    assert out == {
        "doc_id": "doc-race",
        "chunks_count": 1,
        "domain": "rh",
        "already_exists": True,
    }
    assert find_existing.call_count == 2


def test_handle_ingest_raises_when_duplicate_detected_without_recheck_hit():
    with patch("src.ingest.handler.validate_content", return_value="conteudo"), patch(
        "src.ingest.handler.validate_domain", return_value="rh"
    ), patch("src.ingest.handler.compute_content_hash", return_value="d" * 64), patch(
        "src.ingest.handler.chroma_find_by_content_hash", side_effect=[None, None]
    ), patch("src.ingest.handler.chunk_text", return_value=["c1"]), patch(
        "src.ingest.handler.get_embeddings", return_value=[[0.8]]
    ), patch("src.ingest.handler.validate_chunks_and_embeddings"), patch(
        "src.ingest.handler.uuid.uuid4", return_value="33333333-3333-3333-3333-333333333333"
    ), patch(
        "src.ingest.handler.chroma_add", return_value=False
    ):
        with pytest.raises(RuntimeError, match="Conflito de inserção"):
            handle_ingest("conteudo", "rh")
