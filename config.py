# config.py

"""Configuration constants for the RAG pipeline.
Replace paths and parameters as needed for your environment.
"""

import os

# ---- Base directory (repo root) ----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---- LM Studio ----
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

# ---- Paths ----
RAW_DOCS_DIR = os.getenv(
    "RAW_DOCS_DIR",
    os.path.join(BASE_DIR, "data", "books")
)

# Vector store paths (in faiss_store/)
def _resolve_path(primary: str, fallback: str) -> str:
    return primary if os.path.exists(primary) else (fallback if os.path.exists(fallback) else primary)

INDEX_PATH = os.getenv(
    "INDEX_PATH",
    _resolve_path(
        os.path.join(BASE_DIR, "faiss_store", "index.faiss"),
        os.path.join(BASE_DIR, "index.faiss")
    )
)
METADATA_PATH = os.getenv(
    "METADATA_PATH",
    _resolve_path(
        os.path.join(BASE_DIR, "faiss_store", "chunk_store.json"),
        os.path.join(BASE_DIR, "chunk_store.json")
    )
)
BM25_INDEX_PATH = os.getenv(
    "BM25_INDEX_PATH",
    _resolve_path(
        os.path.join(BASE_DIR, "bm25_search", "bm25_index.json"),
        os.path.join(BASE_DIR, "bm25_index.json")
    )
)

EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")

# ---- Chunking ----
CHUNK_SIZE_TOKENS    = 200
CHUNK_OVERLAP_TOKENS = 50
TOKENIZER_NAME       = "simple"

# ---- Embedding model ----
EMBEDDING_PROVIDER   = "local"
EMBEDDING_MODEL_NAME = os.getenv("LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
EMBEDDING_DIM        = 768
EMBED_BATCH_SIZE     = 32

# Number of parallel HTTP workers for embedding.
# 4 workers provides peak throughput (~195 chunks/sec) on RTX 3050.
EMBED_WORKERS: int = int(os.getenv("EMBED_WORKERS", "4"))

# ---- FAISS ----
FAISS_INDEX_TYPE     = "IndexFlatIP"   # cosine via inner product on normalised vecs
NORMALIZE_EMBEDDINGS = True

# ---- Retrieval ----
TOP_K               = 5
DEFAULT_SEARCH_MODE = "hybrid"   # "hybrid" | "semantic" | "keyword"
HYBRID_ALPHA        = 0.5        # semantic weight (0 = pure keyword, 1 = pure semantic)
RRF_K               = 60         # Reciprocal Rank Fusion constant

# ---- Keyword Search (BM25) ----
BM25_K1 = 1.5
BM25_B  = 0.75

# ---- Generation / LLM ----
LLM_PROVIDER          = "local"
LLM_MODEL_NAME        = os.getenv("LM_STUDIO_LLM_MODEL", "qwen/qwen3-4b")
MAX_TOKENS_RESPONSE   = 512
PROMPT_TEMPLATE_PATH  = None   # defined in generation/prompt_template.py

# ---- Eval ----
EVAL_TOP_K = 5
