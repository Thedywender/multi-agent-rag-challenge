from unittest.mock import patch

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_documents_requires_domain():
    resp = client.post("/documents", json={"content": "abc"})
    assert resp.status_code == 422


def test_documents_invalid_domain():
    resp = client.post("/documents", json={"content": "abc", "domain": "invalid"})
    assert resp.status_code == 422


def test_documents_empty_content():
    resp = client.post("/documents", json={"content": "   ", "domain": "rh"})
    assert resp.status_code == 400


def test_ask_success_contract():
    fake = {
        "answer": "ok",
        "sources": [{"document": "doc1", "metadata": {"domain": "rh"}}],
        "routed_domain": "rh",
    }
    with patch("src.main.handle_ask", return_value=fake):
        resp = client.post("/ask", json={"question": "Qual a política de férias?"})

    assert resp.status_code == 200
    assert resp.json() == fake


def test_ask_empty_question():
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["field"] == "question"
    assert body["error"]["code"] == "empty_field"
