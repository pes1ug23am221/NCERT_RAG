# prompt_template.py
#
# Purpose: single source of truth for how retrieved chunks + the user query
# get assembled into the final prompt sent to the LLM.

PROMPT_TEMPLATE = """You are a helpful study assistant. Answer the following question using ONLY the provided context from NCERT textbooks.

Context:
{context}

Question: {question}

Instructions:
- Answer based strictly on the context above.
- Cite the source in brackets after each relevant point, e.g. [Class 12 Physics – Thermodynamics].
- If the answer cannot be found in the context, say "I don't know."
- Be clear and concise.
"""


def _make_citation(chunk: dict) -> str:
    """Build a human-readable citation string from chunk metadata.

    Prefers rich book metadata (class_label + title) when available,
    falls back to source_filename.
    """
    class_label = chunk.get("class_label", "")
    title = chunk.get("title", "")
    subject = chunk.get("subject", "")
    source = chunk.get("source_filename", "")

    if class_label and title:
        return f"{class_label} {subject} \u2013 {title}"
    elif class_label and subject:
        return f"{class_label} {subject}"
    elif source:
        return source
    return "Unknown source"


def format_context(chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single context string.

    Each chunk dict is expected to contain at least:
        - "chunk_text": the text of the chunk
    Optional rich metadata (class_label, subject, title, source_filename)
    is used to build a clear citation label.
    """
    formatted_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("chunk_text", "")
        citation = _make_citation(chunk)
        formatted_chunks.append(f"[{i}] {text}\n    Source: {citation}")
    return "\n\n".join(formatted_chunks)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Create the final prompt by injecting the formatted context and the query."""
    context_str = format_context(chunks)
    return PROMPT_TEMPLATE.format(context=context_str, question=query)
