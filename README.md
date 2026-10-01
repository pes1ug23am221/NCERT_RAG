# NCERT RAG Pipeline

A hybrid RAG (Retrieval-Augmented Generation) system built over NCERT textbooks. Given a question, it retrieves relevant passages using both semantic search (FAISS) and keyword search (BM25), fuses the results, and generates an answer via a local LLM running in LM Studio.

---

## How it works

```
Query
  │
  ├─► BM25 keyword search  ──┐
  │                           ├─► Reciprocal Rank Fusion ─► Top-K chunks ─► LLM ─► Answer
  └─► FAISS semantic search ─┘
```

1. **PDF processing** — raw NCERT PDFs are extracted to plain text using `pypdf`, cleaned, and cached to disk.
2. **Chunking** — text is split into 200-token windows with 50-token overlap using a sentence-aware splitter.
3. **Embeddings** — chunks are embedded in parallel (4 workers) via the `nomic-embed-text-v1.5` model served by LM Studio.
4. **FAISS index** — 768-dimensional vectors stored in a `IndexFlatIP` (cosine similarity via inner product on normalised vectors).
5. **BM25 index** — tokenised chunks indexed with BM25Okapi (`k1=1.5`, `b=0.75`) for keyword recall.
6. **Hybrid retrieval** — both indexes are queried and their ranked lists are fused with Reciprocal Rank Fusion (RRF, `k=60`).
7. **Generation** — top chunks are packed into a prompt and sent to a local LLM (`qwen/qwen3-4b`) via LM Studio.

---

## Project structure

```
.
├── config.py               # All settings: paths, model names, chunk size, retrieval params
├── rag_pipeline.py          # Entry point — runs a query end to end
├── requirements.txt
│
├── pdf_processing/          # PDF extraction (ProcessPoolExecutor, disk-cached)
├── chunking/                # Sentence-aware sliding window chunker
├── embeddings/              # Parallel embedding with checkpointing
├── faiss_store/             # FAISS index + chunk metadata store
├── bm25_search/             # BM25 keyword index
└── generation/              # Prompt builder + LM Studio LLM call
```

---

## Setup

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai) running locally with:
  - An embedding model loaded: `text-embedding-nomic-embed-text-v1.5`
  - A chat model loaded: `qwen/qwen3-4b` (or any compatible model)
- NCERT PDFs placed under `data/books/`

### Install

```bash
pip install -r requirements.txt
```

### Build the indexes (first run only)

```bash
python rag_pipeline.py --rebuild
```

This extracts text from all PDFs, chunks them, embeds and indexes them. It resumes safely if interrupted.

### Query

```bash
python rag_pipeline.py "What is Newton's second law?"
```

---

## Configuration

All tunable parameters live in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio server URL |
| `CHUNK_SIZE_TOKENS` | `200` | Tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `EMBED_WORKERS` | `4` | Parallel embedding threads |
| `TOP_K` | `5` | Chunks returned to the LLM |
| `HYBRID_ALPHA` | `0.5` | Semantic vs keyword weight |
| `LLM_MODEL_NAME` | `qwen/qwen3-4b` | Chat model name in LM Studio |

---

## Data

The `data/books/` folder and built indexes (`faiss_store/index.faiss`, `bm25_search/bm25_index.json`, `faiss_store/chunk_store.json`) are not tracked in git. Run `--rebuild` after adding your PDFs.
