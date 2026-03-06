"""Divisão de texto em chunks com overlap."""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide o texto em chunks com overlap.

    Args:
        text: Texto a ser dividido.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        overlap: Número de caracteres de sobreposição entre chunks.

    Returns:
        Lista de chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                end = start + last_space + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if overlap < end - start else end

    return chunks
