# keyword_search.py
"""BM25 keyword search implementation for the RAG pipeline.

Provides fast lexical search over chunked documents using BM25Okapi scoring
with an inverted index for sub-millisecond retrieval.
"""

import os
import re
import math
import json
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import config

# Module-level state
_bm25_index: Optional["BM25Index"] = None
_chunks: List[Dict] = []


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """BM25Okapi index with inverted index for fast keyword retrieval."""

    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_len: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        # Inverted index: term -> list of (doc_idx, term_freq)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def fit(self, chunks: List[Dict]):
        """Build the BM25 inverted index from the provided chunks."""
        self.corpus_size = len(chunks)
        self.doc_len = []
        self.doc_freqs = defaultdict(int)
        self.inverted_index = defaultdict(list)

        total_length = 0
        for doc_idx, chunk in enumerate(chunks):
            text = chunk.get("chunk_text", "")
            tokens = tokenize(text)
            length = len(tokens)
            self.doc_len.append(length)
            total_length += length

            counts = Counter(tokens)
            for term, count in counts.items():
                self.inverted_index[term].append((doc_idx, count))
                self.doc_freqs[term] += 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0
        # Convert defaultdicts to regular dicts for cleaner inspection / serialization
        self.doc_freqs = dict(self.doc_freqs)
        self.inverted_index = dict(self.inverted_index)

    def score(self, query: str) -> Dict[int, float]:
        """Compute BM25Okapi scores for all matching documents."""
        q_tokens = tokenize(query)
        if not q_tokens or self.corpus_size == 0:
            return {}

        scores: Dict[int, float] = defaultdict(float)
        N = self.corpus_size
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for term in q_tokens:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            # Robertson-Spärck Jones IDF with smoothing
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            postings = self.inverted_index.get(term, [])
            for doc_idx, freq in postings:
                dl = self.doc_len[doc_idx]
                denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
                term_score = idf * ((freq * (k1 + 1.0)) / denom)
                scores[doc_idx] += term_score

        return dict(scores)

    def search(self, chunks: List[Dict], query: str, k: int = config.TOP_K) -> List[Dict]:
        """Search chunks using BM25 and return top-k results with metadata."""
        scores = self.score(query)
        if not scores:
            return []

        # Sort doc_idx by score descending
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results: List[Dict] = []
        for doc_idx, sc in top_items:
            if 0 <= doc_idx < len(chunks):
                meta = chunks[doc_idx]
                results.append(
                    {
                        "chunk_text": meta.get("chunk_text", ""),
                        "source_filename": meta.get("source_filename", ""),
                        "chunk_id": meta.get("chunk_id", ""),
                        "doc_idx": doc_idx,
                        "score": float(sc),
                        "search_type": "keyword",
                    }
                )
        return results

    def to_dict(self) -> dict:
        """Serialize index structures to a serializable dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "doc_freqs": self.doc_freqs,
            "inverted_index": self.inverted_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        """Reconstruct index from a serialized dictionary."""
        idx = cls(k1=data.get("k1", config.BM25_K1), b=data.get("b", config.BM25_B))
        idx.corpus_size = data.get("corpus_size", 0)
        idx.avgdl = data.get("avgdl", 0.0)
        idx.doc_len = data.get("doc_len", [])
        idx.doc_freqs = data.get("doc_freqs", {})
        # Note: JSON serializes tuples as lists, convert to tuple
        idx.inverted_index = {
            term: [tuple(p) for p in postings]
            for term, postings in data.get("inverted_index", {}).items()
        }
        return idx


# ── Module-level API ─────────────────────────────────────────────────────────

def create_index(chunks: List[Dict]):
    """Build and set the active in-memory BM25 index."""
    global _bm25_index, _chunks
    _chunks = chunks
    index = BM25Index()
    index.fit(chunks)
    _bm25_index = index


def is_ready() -> bool:
    """Check if BM25 index is loaded and ready."""
    return _bm25_index is not None and len(_chunks) > 0


def set_chunks(chunks: List[Dict]):
    """Set the referenced chunks list."""
    global _chunks
    _chunks = chunks


def get_chunks() -> List[Dict]:
    """Get the current chunks list."""
    return _chunks


def search(query: str, k: int = config.TOP_K) -> List[Dict]:
    """Execute keyword search against the active BM25 index."""
    if _bm25_index is None:
        raise RuntimeError("BM25 index not initialized. Call create_index or load first.")
    return _bm25_index.search(_chunks, query, k=k)


def save(path: str = config.BM25_INDEX_PATH):
    """Save the active BM25 index to a JSON file."""
    if _bm25_index is None:
        raise RuntimeError("No active BM25 index to save.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_bm25_index.to_dict(), f, ensure_ascii=False)


def load(path: str = config.BM25_INDEX_PATH, chunks: Optional[List[Dict]] = None):
    """Load BM25 index from a JSON file."""
    global _bm25_index, _chunks
    if not os.path.exists(path):
        raise RuntimeError(f"BM25 index file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _bm25_index = BM25Index.from_dict(data)
    if chunks is not None:
        _chunks = chunks


def index_exists(path: str = config.BM25_INDEX_PATH) -> bool:
    """Return True if the BM25 index file exists."""
    return os.path.exists(path)
