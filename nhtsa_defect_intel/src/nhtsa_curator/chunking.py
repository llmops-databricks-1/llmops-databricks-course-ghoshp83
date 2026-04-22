"""Text chunker for narrative + parsed-document content.

We use a simple character-based sliding window with a preferred split
on a separator (defaults to ``\\n\\n`` paragraph break). This is the
same shape the reference arxiv-curator uses; it keeps the index build
deterministic and avoids pulling in a heavy dependency like
``langchain_text_splitters`` for what is essentially three lines of
real logic.

Why character-based and not token-based: NHTSA narratives are short
and bursty. Token estimates within ±25 % of char count are good
enough for retrieval, and char-based splitting means we can chunk
without instantiating a tokenizer per row in Spark.

Algorithm:
    1. If the text is shorter than ``chunk_size``, return ``[text]``.
    2. Slide a window of ``chunk_size`` chars; after each window,
       advance by ``chunk_size - chunk_overlap``.
    3. Within each window, prefer to break at the *last* occurrence of
       ``separator`` (so paragraphs stay intact). Fall back to a hard
       cut if no separator exists in-window.
"""

from __future__ import annotations

from collections.abc import Iterable


def chunk_text(
    text: str | None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separator: str = "\n\n",
) -> list[str]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: source text. ``None`` / empty returns ``[]``.
        chunk_size: max characters per chunk.
        chunk_overlap: overlap between consecutive chunks (carries
            context across boundary).
        separator: preferred soft-break string within a window.

    Returns:
        Ordered list of chunk strings (no chunk is empty, none exceeds
        ``chunk_size``).
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    stride = chunk_size - chunk_overlap
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        # Prefer the last separator within the window so paragraphs survive.
        if end < n and separator:
            window = text[i:end]
            cut = window.rfind(separator)
            if cut > 0:  # require non-degenerate split
                end = i + cut + len(separator)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(end - chunk_overlap, i + 1)
    return chunks


def chunk_records(
    records: Iterable[dict],
    text_field: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separator: str = "\n\n",
) -> list[dict]:
    """Apply :func:`chunk_text` over an iterable of dict records.

    Each input record is fanned out into 1..N output records, each
    carrying the original metadata plus ``chunk_idx`` and ``content``.
    Records whose ``text_field`` is empty are dropped.
    """
    out: list[dict] = []
    for rec in records:
        body = rec.get(text_field) or ""
        chunks = chunk_text(body, chunk_size, chunk_overlap, separator)
        for idx, chunk in enumerate(chunks):
            new = {k: v for k, v in rec.items() if k != text_field}
            new["chunk_idx"] = idx
            new["content"] = chunk
            out.append(new)
    return out
