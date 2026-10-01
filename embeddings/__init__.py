# embeddings/__init__.py
"""Dense vector embedding unit using LM Studio API."""

from .embeddings import (
    embed_texts,
    embed_query,
)

__all__ = [
    "embed_texts",
    "embed_query",
]
