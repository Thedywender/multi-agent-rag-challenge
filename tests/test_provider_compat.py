import pytest

from src.shared.embeddings import _get_provider as emb_provider
from src.shared.llm import _get_provider as llm_provider


@pytest.mark.parametrize("value", ["openai", "bedrock"])
def test_valid_provider(monkeypatch, value):
    monkeypatch.setenv("LLM_PROVIDER", value)
    assert emb_provider() == value
    assert llm_provider() == value


def test_invalid_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "xpto")
    with pytest.raises(ValueError):
        emb_provider()
    with pytest.raises(ValueError):
        llm_provider()
