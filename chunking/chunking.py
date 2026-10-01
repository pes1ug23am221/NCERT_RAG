# chunking.py
"""Utility functions to split cleaned document text into overlapping chunks.
The implementation is deliberately simple and does **not** rely on an external
tokenizer – we approximate token count by splitting on whitespace.  This works
well enough for the prototype and keeps the dependency footprint low.
"""

import re
from typing import List, Dict

import config

# Metadata fields that should be propagated from document to each chunk
_METADATA_FIELDS = ("class", "class_label", "subject", "title", "stream", "total_pages")


def count_tokens(text: str) -> int:
    """Return an approximate token count for *text*.

    For the purpose of this pipeline we treat a token as a whitespace-separated
    word.  This is sufficient because the chunk size parameters are modest
    (200 tokens) and the downstream embedding model tokenizer handles the exact
    tokenisation.
    """
    return len(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Split *text* into sentences using a simple regular expression.

    The regex looks for a period, exclamation mark, or question mark followed
    by whitespace and a capital letter – a heuristic that works for most
    English prose.  Newlines are also treated as sentence boundaries.
    """
    normalized = text.replace("\n", " ")
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, normalized)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_from_sentences(
    sentences: List[str], chunk_size_tokens: int, overlap_tokens: int
) -> List[str]:
    """Create chunks from a list of *sentences*.

    The algorithm accumulates sentences until the token count reaches
    ``chunk_size_tokens``.  When a chunk is emitted, the next chunk starts with
    the last ``overlap_tokens`` tokens from the previous chunk (implemented by
    joining the trailing part of the previous chunk's text).
    """
    chunks: List[str] = []
    current: List[str] = []
    current_token_len = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        # If a single sentence exceeds the chunk size, emit it directly.
        if sentence_tokens > chunk_size_tokens and not current:
            chunks.append(sentence)
            continue

        if current_token_len + sentence_tokens <= chunk_size_tokens:
            current.append(sentence)
            current_token_len += sentence_tokens
        else:
            # Emit current chunk.
            chunks.append(" ".join(current))
            # Prepare overlap.
            overlap_text = " ".join(current)
            overlap_words = overlap_text.split()
            overlap_tail = " ".join(overlap_words[-overlap_tokens:]) if overlap_tokens > 0 else ""
            # Start new chunk with overlap + current sentence.
            current = []
            if overlap_tail:
                current.append(overlap_tail)
                current_token_len = count_tokens(overlap_tail)
            else:
                current_token_len = 0
            current.append(sentence)
            current_token_len += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc_text: str,
    chunk_size_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split a single document's text into overlapping chunks.

    Returns a list of chunk strings.
    """
    sentences = split_into_sentences(doc_text)
    return _chunk_from_sentences(sentences, chunk_size_tokens, overlap_tokens)


def chunk_all_documents(
    documents: List[Dict],
    chunk_size_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> List[Dict]:
    """Chunk *documents* and attach metadata.

    Each input document is expected to be a dictionary with the keys
    ``doc_id``, ``source_filename`` and ``text`` (as produced by
    :func:`ingest.load_all_documents`).  Optional book-level metadata keys
    (``class``, ``class_label``, ``subject``, ``title``, ``stream``,
    ``total_pages``) are forwarded to every chunk when present.

    The function returns a flat list of chunk dicts containing:

    - ``doc_id``          – identifier of the source document.
    - ``chunk_id``        – unique identifier for the chunk (``{doc_id}_{idx}``).
    - ``source_filename`` – original file name for citation.
    - ``chunk_text``      – the text of the chunk.
    - ``position``        – integer index of the chunk within its document.
    - ``class``           – (NCERT) class number, e.g. 12.
    - ``class_label``     – (NCERT) e.g. "Class 12".
    - ``subject``         – (NCERT) e.g. "Physics".
    - ``title``           – (NCERT) book title, e.g. "Physics Part-I".
    - ``stream``          – (NCERT) stream category, e.g. "Science".
    - ``total_pages``     – total pages in the source PDF.
    """
    all_chunks: List[Dict] = []
    for doc in documents:
        doc_id = doc["doc_id"]
        source = doc["source_filename"]
        text = doc["text"]
        chunks = chunk_document(text, chunk_size_tokens, overlap_tokens)
        for idx, chunk_text in enumerate(chunks):
            chunk: Dict = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx}",
                "source_filename": source,
                "chunk_text": chunk_text,
                "position": idx,
            }
            # Propagate book-level metadata fields if present in the document
            for field in _METADATA_FIELDS:
                if field in doc:
                    chunk[field] = doc[field]
            all_chunks.append(chunk)
    return all_chunks


if __name__ == "__main__":  # pragma: no cover
    sample = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "unlike the natural intelligence displayed by humans and animals. "
        "AI research is defined as the study of intelligent agents."
    )
    chunks = chunk_document(sample, 10, 2)
    print("Chunks:")
    for c in chunks:
        print(f"- ({count_tokens(c)} tokens) {c}")
