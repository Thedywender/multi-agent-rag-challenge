"""Validações reutilizáveis para ingestão e persistência."""

from __future__ import annotations

import hashlib
import uuid
import math
from numbers import Real


class ValidationInputError(ValueError):
    __slots__ = ("field", "code", "message")

    def __init__(
        self,
        message: str,
        *,
        field: str,
        code: str = "validation_error",
    ) -> None:
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


ALLOWED_DOMAINS = frozenset({"rh", "tecnico"})


def validate_non_empty_string(
    value: object,
    field_name: str,
    *,
    normalize: bool = True,
    lower: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValidationInputError(
            f"O campo '{field_name}' deve ser string.",
            field=field_name,
            code="invalid_type",
        )

    stripped = value.strip()
    if not stripped:
        raise ValidationInputError(
            f"O campo '{field_name}' não pode estar vazio.",
            field=field_name,
            code="empty_field",
        )

    result = stripped if normalize else value
    return result.lower() if lower else result


def validate_content(content: object) -> str:
    return validate_non_empty_string(
        content,
        "content",
        normalize=False,
        lower=False,
    )


def validate_domain(domain: object) -> str:
    normalized = validate_non_empty_string(
        domain,
        "domain",
        normalize=True,
        lower=True,
    )

    if normalized not in ALLOWED_DOMAINS:
        allowed = ", ".join(sorted(ALLOWED_DOMAINS))
        raise ValidationInputError(
            f"Domain inválido. Valores aceitos: {allowed}.",
            field="domain",
            code="invalid_domain",
        )

    return normalized


def validate_question(question: object) -> str:
    return validate_non_empty_string(
        question,
        "question",
        normalize=True,
        lower=False,
    )


def compute_content_hash(content: object) -> str:
    validated_content = validate_content(content)
    return hashlib.sha256(validated_content.encode("utf-8")).hexdigest()


def validate_doc_id(doc_id: object) -> str:
    normalized = validate_non_empty_string(doc_id, "doc_id", normalize=True)

    try:
        uuid.UUID(normalized)
    except ValueError as exc:
        raise ValidationInputError(
            "O campo 'doc_id' deve ser um UUID válido.",
            field="doc_id",
            code="invalid_doc_id",
        ) from exc

    return normalized


def validate_content_hash(content_hash: object, *, strict: bool = False) -> str:
    normalized = validate_non_empty_string(
        content_hash,
        "content_hash",
        normalize=True,
        lower=True,
    )

    if strict:
        if len(normalized) != 64:
            raise ValidationInputError(
                "O campo 'content_hash' deve ter 64 caracteres hexadecimais.",
                field="content_hash",
                code="invalid_hash_length",
            )
        try:
            int(normalized, 16)
        except ValueError as exc:
            raise ValidationInputError(
                "O campo 'content_hash' deve conter apenas caracteres hexadecimais.",
                field="content_hash",
                code="invalid_hash_format",
            ) from exc

    return normalized


def validate_chunks_and_embeddings(chunks: object, embeddings: object) -> None:
    if not isinstance(chunks, list):
        raise ValidationInputError(
            "O campo 'chunks' deve ser uma lista.",
            field="chunks",
            code="invalid_chunks_type",
        )
    if not chunks:
        raise ValidationInputError(
            "O campo 'chunks' não pode ser vazio.",
            field="chunks",
            code="empty_chunks",
        )

    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, str):
            raise ValidationInputError(
                f"O chunk no índice {chunk_index} deve ser string.",
                field="chunks",
                code="invalid_chunk_type",
            )
        if not chunk.strip():
            raise ValidationInputError(
                f"O chunk no índice {chunk_index} não pode ser vazio.",
                field="chunks",
                code="empty_chunk",
            )

    if not isinstance(embeddings, list):
        raise ValidationInputError(
            "O campo 'embeddings' deve ser uma lista.",
            field="embeddings",
            code="invalid_embeddings_type",
        )
    if not embeddings:
        raise ValidationInputError(
            "O campo 'embeddings' não pode ser vazio.",
            field="embeddings",
            code="empty_embeddings",
        )
    if len(chunks) != len(embeddings):
        raise ValidationInputError(
            "A quantidade de 'embeddings' deve ser igual à quantidade de 'chunks'.",
            field="embeddings",
            code="mismatched_lengths",
        )

    expected_dim: int | None = None
    for emb_index, embedding in enumerate(embeddings):
        if not isinstance(embedding, list):
            raise ValidationInputError(
                f"O embedding no índice {emb_index} deve ser uma lista.",
                field="embeddings",
                code="invalid_embedding_type",
            )
        if not embedding:
            raise ValidationInputError(
                f"O embedding no índice {emb_index} não pode ser vazio.",
                field="embeddings",
                code="empty_embedding",
            )

        if expected_dim is None:
            expected_dim = len(embedding)
        elif len(embedding) != expected_dim:
            raise ValidationInputError(
                "Todos os embeddings devem ter a mesma dimensão.",
                field="embeddings",
                code="inconsistent_embedding_dimension",
            )

        for value_index, value in enumerate(embedding):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValidationInputError(
                    f"O valor {value_index} do embedding {emb_index} deve ser numérico.",
                    field="embeddings",
                    code="invalid_embedding_value_type",
                )
            if not math.isfinite(float(value)):
                raise ValidationInputError(
                    f"O valor {value_index} do embedding {emb_index} deve ser finito.",
                    field="embeddings",
                    code="invalid_embedding_value",
                )


def validate_query_embedding(query_embedding: object) -> list[float]:
    if not isinstance(query_embedding, list):
        raise ValidationInputError(
            "O campo 'query_embedding' deve ser uma lista.",
            field="query_embedding",
            code="invalid_query_embedding_type",
        )

    if not query_embedding:
        raise ValidationInputError(
            "O campo 'query_embedding' não pode ser vazio.",
            field="query_embedding",
            code="empty_query_embedding",
        )

    normalized: list[float] = []
    for value_index, value in enumerate(query_embedding):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationInputError(
                f"O valor {value_index} de 'query_embedding' deve ser numérico.",
                field="query_embedding",
                code="invalid_query_embedding_value_type",
            )

        number = float(value)
        if not math.isfinite(number):
            raise ValidationInputError(
                f"O valor {value_index} de 'query_embedding' deve ser finito.",
                field="query_embedding",
                code="invalid_query_embedding_value",
            )
        normalized.append(number)

    return normalized


def validate_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationInputError(
            "O campo 'k' deve ser inteiro.",
            field="k",
            code="invalid_k_type",
        )
    if k <= 0:
        raise ValidationInputError(
            "O campo 'k' deve ser maior que zero.",
            field="k",
            code="invalid_k",
        )
    return k
