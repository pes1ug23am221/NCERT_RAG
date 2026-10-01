# vector_store.py
"""FAISS index management and metadata storage for the RAG pipeline.
The module maintains two module‑level globals:
- ``_index`` – the FAISS index instance.
- ``_chunk_store`` – a list of metadata dicts, where the list position matches the
  internal ID used by FAISS.
All helper functions operate on these globals.
"""

import os
import json
from typing import List, Dict

import numpy as np
import faiss

import config

# Module‑level state (initialized lazily).
_index = None  # type: ignore
_chunk_store: List[Dict] = []


def create_index(dimension: int):
    """Create a FAISS index according to ``config.FAISS_INDEX_TYPE``.

    Supported types:
    - ``IndexFlatIP`` – inner product (used with normalized vectors for cosine).
    - ``IndexFlatL2`` – Euclidean distance.
    """
    global _index
    if config.FAISS_INDEX_TYPE == "IndexFlatIP":
        _index = faiss.IndexFlatIP(dimension)
    elif config.FAISS_INDEX_TYPE == "IndexFlatL2":
        _index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported FAISS_INDEX_TYPE: {config.FAISS_INDEX_TYPE}")
    # If we want to enable ID mapping later, we could wrap with IndexIDMap, but for
    # a flat index the internal integer IDs are sufficient.


def add_chunks(chunks: List[Dict], embeddings: np.ndarray):
    """Add *embeddings* to the FAISS index and store corresponding *chunks*.

    ``chunks`` must be in the same order as the rows of ``embeddings``.
    ``embeddings`` is expected to be a NumPy ``float32`` array of shape
    ``(n_chunks, dim)``.
    """
    global _index, _chunk_store
    if _index is None:
        raise RuntimeError("FAISS index has not been created. Call create_index first.")
    if config.NORMALIZE_EMBEDDINGS:
        faiss.normalize_L2(embeddings)
    # Add vectors to index.
    _index.add(embeddings)
    # Extend metadata store – FAISS IDs are 0‑based sequential, matching list order.
    _chunk_store.extend(chunks)


def search(query_embedding: np.ndarray, k: int = config.TOP_K) -> List[Dict]:
    """Search the index with *query_embedding* and return the top‑k chunks.

    The result dict contains ``chunk_text``, ``source_filename``, ``chunk_id``
    and ``score`` (higher is better for inner‑product similarity).
    """
    if _index is None:
        raise RuntimeError("FAISS index not loaded or created.")
    if config.NORMALIZE_EMBEDDINGS:
        faiss.normalize_L2(query_embedding)
    # Ensure query is 2‑D (FAISS expects shape (1, dim)).
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]
    distances, indices = _index.search(query_embedding, k)
    results: List[Dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_chunk_store):
            continue
        meta = _chunk_store[idx]
        results.append(
            {
                "chunk_text": meta["chunk_text"],
                "source_filename": meta["source_filename"],
                "chunk_id": meta["chunk_id"],
                "doc_idx": int(idx),
                "score": float(dist),
                "search_type": "semantic",
            }
        )
    return results


def get_chunks() -> List[Dict]:
    """Return all stored chunk metadata dictionaries."""
    return _chunk_store


def save(index_path: str = config.INDEX_PATH, metadata_path: str = config.METADATA_PATH):
    """Persist the FAISS index and the metadata store to disk.
    """
    if _index is None:
        raise RuntimeError("No FAISS index to save.")
    # Write the index binary.
    faiss.write_index(_index, index_path)
    # Store metadata as JSON (ensuring UTF‑8 safe).
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_chunk_store, f, ensure_ascii=False, indent=2)


def load(index_path: str = config.INDEX_PATH, metadata_path: str = config.METADATA_PATH):
    """Load a previously saved FAISS index and its metadata.
    """
    global _index, _chunk_store
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise RuntimeError("FAISS index or metadata file not found.")
    _index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        _chunk_store = json.load(f)


def index_exists(index_path: str = config.INDEX_PATH, metadata_path: str = config.METADATA_PATH) -> bool:
    """Return ``True`` if both the index and its metadata files exist.
    """
    return os.path.exists(index_path) and os.path.exists(metadata_path)


if __name__ == "__main__":  # pragma: no cover
    # Tiny manual test – create a dummy index, add two fake chunks, search.
    dim = config.EMBEDDING_DIM
    create_index(dim)
    dummy_chunks = [
        {"doc_id": "doc1", "chunk_id": "doc1_0", "source_filename": "a.txt", "chunk_text": "Hello world", "position": 0},
        {"doc_id": "doc2", "chunk_id": "doc2_0", "source_filename": "b.txt", "chunk_text": "Goodbye world", "position": 0},
    ]
    dummy_emb = np.random.rand(2, dim).astype(np.float32)
    add_chunks(dummy_chunks, dummy_emb)
    q = np.random.rand(1, dim).astype(np.float32)
    print("Search results:", search(q, k=2))
