import json
from unittest.mock import patch

import pytest
from fastapi import Request
from pydantic import ValidationError

from src.main import (
    AskRequest,
    AskResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    health,
    post_ask,
    post_documents,
    unexpected_error_handler,
    value_error_handler,
)
from src.shared.validators import ValidationInputError


def _make_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
        }
    )


def test_health_contract():
    resp = health()
    assert resp.status == "ok"


def test_document_request_requires_domain():
    with pytest.raises(ValidationError):
        DocumentRequest(content="abc")


def test_document_request_invalid_domain():
    with pytest.raises(ValidationError):
        DocumentRequest(content="abc", domain="invalid")


def test_post_documents_success_contract():
    fake = {
        "doc_id": "11111111-1111-1111-1111-111111111111",
        "chunks_count": 2,
        "domain": "rh",
        "already_exists": False,
    }

    with patch("src.main.handle_ingest", return_value=fake):
        response = post_documents(DocumentRequest(content="abc", domain="rh"))

    assert isinstance(response, DocumentResponse)
    assert response.model_dump() == fake


def test_post_ask_success_contract():
    fake = {
        "answer": "ok",
        "sources": [{"document": "doc1", "metadata": {"domain": "rh"}}],
        "routed_domain": "rh",
    }

    with patch("src.main.handle_ask", return_value=fake):
        response = post_ask(AskRequest(question="Qual a política de férias?"))

    assert isinstance(response, AskResponse)
    assert response.model_dump() == fake


def test_value_error_handler_for_validation_input_error():
    exc = ValidationInputError(
        "O campo 'question' não pode estar vazio.",
        field="question",
        code="empty_field",
    )

    response = value_error_handler(_make_request("/ask"), exc)
    body = json.loads(response.body)

    assert response.status_code == 400
    parsed = ErrorResponse.model_validate(body)
    assert parsed.error.code == "empty_field"
    assert parsed.error.field == "question"


def test_value_error_handler_for_generic_value_error():
    response = value_error_handler(_make_request("/ask"), ValueError("falha externa"))
    body = json.loads(response.body)

    assert response.status_code == 503
    parsed = ErrorResponse.model_validate(body)
    assert parsed.error.code == "service_unavailable"
    assert parsed.error.message == "falha externa"


def test_unexpected_error_handler_contract():
    response = unexpected_error_handler(_make_request("/ask"), RuntimeError("boom"))
    body = json.loads(response.body)

    assert response.status_code == 500
    parsed = ErrorResponse.model_validate(body)
    assert parsed.error.code == "internal_error"
    assert parsed.error.field is None
