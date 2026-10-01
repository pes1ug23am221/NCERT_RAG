# faiss_store/__init__.py
"""FAISS vector storage and similarity search unit."""

from .vector_store import (
    create_index,
    add_chunks,
    save,
    load,
    search,
    get_chunks,
    index_exists,
)

__all__ = [
    "create_index",
    "add_chunks",
    "save",
    "load",
    "search",
    "get_chunks",
    "index_exists",
]
