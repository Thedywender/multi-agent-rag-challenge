from unittest.mock import patch

import pytest

from src.query.handler import handle_ask
from src.shared.validators import ValidationInputError


def test_handle_ask_validates_question_then_calls_langgraph():
    with patch("src.query.handler.validate_question", return_value="Pergunta válida") as validate_question, patch(
        "src.query.handler.ask_with_langgraph",
        return_value={"answer": "ok", "sources": [], "routed_domain": "rh"},
    ) as ask_with_langgraph:
        out = handle_ask("   Pergunta válida   ", k=4)

    assert out == {"answer": "ok", "sources": [], "routed_domain": "rh"}
    validate_question.assert_called_once_with("   Pergunta válida   ")
    ask_with_langgraph.assert_called_once_with("Pergunta válida", k=4)


def test_handle_ask_propagates_validation_input_error():
    error = ValidationInputError(
        "O campo 'question' não pode estar vazio.",
        field="question",
        code="empty_field",
    )

    with patch("src.query.handler.validate_question", side_effect=error):
        with pytest.raises(ValidationInputError) as exc:
            handle_ask("   ")

    assert exc.value.field == "question"
    assert exc.value.code == "empty_field"
