# rag_pipeline.py
"""Top-level orchestrator for the local RAG pipeline.

Ties together ingest, chunking, embedding, FAISS storage, and LLM answer
generation.  Run directly with ``python rag_pipeline.py``.
"""

import os
import sys
import time
import numpy as np

import config
import pdf_processing as ingest
import chunking
import embeddings as emb_mod
import faiss_store as vector_store
import bm25_search as keyword_search
from generation import prompt_template, generate


# ── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    """Print a message, replacing characters unsupported by the current console."""
    print(msg.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))


def verify_models():
    """Verify that both LM Studio models respond before any heavy work."""
    _print("[CHECK] Verifying embedding model...")
    try:
        emb_mod.embed_texts(["ping"])
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model '{config.EMBEDDING_MODEL_NAME}' is not reachable.\n"
            f"  -> Make sure LM Studio is running and the model is loaded.\n"
            f"  -> Error: {exc}"
        ) from exc
    _print(f"  OK  Embedding model: {config.EMBEDDING_MODEL_NAME}")

    _print("[CHECK] Verifying LLM model...")
    try:
        generate.generate_answer("ping")
    except Exception as exc:
        raise RuntimeError(
            f"LLM model '{config.LLM_MODEL_NAME}' is not reachable.\n"
            f"  -> Make sure LM Studio is running and the model is loaded.\n"
            f"  -> Error: {exc}"
        ) from exc
    _print(f"  OK  LLM model: {config.LLM_MODEL_NAME}")


# ── index management ──────────────────────────────────────────────────────────

def build_index(reuse_embeddings: bool = True):
    """Build the FAISS semantic index and BM25 keyword index from raw documents."""
    t_total = time.time()

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    _print("\n" + "=" * 62)
    _print("  STAGE 1 / 4  |  INGESTING DOCUMENTS")
    _print("=" * 62)
    t1 = time.time()
    documents = ingest.load_all_documents(config.RAW_DOCS_DIR)
    if not documents:
        raise RuntimeError(
            f"No documents found in {config.RAW_DOCS_DIR}. "
            "Add .txt, .md or .pdf files and try again."
        )
    total_chars = sum(len(d["text"]) for d in documents)
    _print(
        f"\n  [OK] {len(documents)} docs loaded | {total_chars:,} chars"
        f" | {time.time()-t1:.1f}s"
    )

    # ── Stage 2: Chunk ───────────────────────────────────────────────────────
    _print("\n" + "=" * 62)
    _print("  STAGE 2 / 4  |  CHUNKING")
    _print("=" * 62)
    t2 = time.time()
    chunks = chunking.chunk_all_documents(
        documents, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
    )
    avg_per_doc = len(chunks) // max(len(documents), 1)
    _print(
        f"  [OK] {len(chunks):,} chunks created"
        f" | avg {avg_per_doc} chunks/doc"
        f" | {time.time()-t2:.1f}s"
    )

    # ── Stage 3: Embed ───────────────────────────────────────────────────────
    _print("\n" + "=" * 62)
    _print(f"  STAGE 3 / 4  |  EMBEDDING  ({config.EMBED_WORKERS} workers, parallel)")
    _print("=" * 62)
    t3 = time.time()
    texts = [c["chunk_text"] for c in chunks]

    embeddings_array = None
    cache_path = getattr(config, "EMBEDDINGS_CACHE_PATH", None)
    if reuse_embeddings and cache_path and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if cached.shape == (len(texts), config.EMBEDDING_DIM):
                embeddings_array = cached
                _print(f"  [OK] Loaded precomputed embeddings from {cache_path} ({cached.shape})")
        except Exception:
            pass

    if embeddings_array is None:
        embeddings_array = emb_mod.embed_texts(
            texts,
            batch_size=getattr(config, "EMBED_BATCH_SIZE", 32),
            workers=config.EMBED_WORKERS,
        )
        if cache_path:
            try:
                np.save(cache_path, embeddings_array)
                _print(f"  [CACHE] Saved embeddings array -> {cache_path}")
            except Exception as e:
                _print(f"  [WARN] Failed to save embeddings cache: {e}")

    embed_elapsed = time.time() - t3
    _print(
        f"  [OK] Shape: {embeddings_array.shape}"
        f" | {embed_elapsed:.1f}s"
        f" | {len(texts)/max(embed_elapsed, 0.001):.1f} chunks/s"
    )

    # ── Stage 4: Build indices ───────────────────────────────────────────────
    _print("\n" + "=" * 62)
    _print("  STAGE 4 / 4  |  BUILDING INDICES")
    _print("=" * 62)
    t4 = time.time()

    _print("  Building FAISS semantic index...")
    vector_store.create_index(config.EMBEDDING_DIM)
    vector_store.add_chunks(chunks, embeddings_array)
    vector_store.save(config.INDEX_PATH, config.METADATA_PATH)
    _print(f"  [OK] FAISS saved -> {config.INDEX_PATH}")

    _print("  Building BM25 keyword index...")
    keyword_search.create_index(chunks)
    keyword_search.save(config.BM25_INDEX_PATH)
    _print(f"  [OK] BM25  saved -> {config.BM25_INDEX_PATH}")

    _print(f"  Index build: {time.time()-t4:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    _print("\n" + "=" * 62)
    _print(f"  DONE  |  Total time: {time.time()-t_total:.1f}s")
    _print(f"         {len(documents)} docs | {len(chunks):,} chunks | {embeddings_array.shape[1]}d vectors")
    _print("=" * 62 + "\n")


def load_index_if_exists() -> bool:
    """Load persisted FAISS and BM25 indices if available. Returns True on success."""
    if not vector_store.index_exists(config.INDEX_PATH, config.METADATA_PATH):
        return False

    _print("[INDEX] Loading existing FAISS semantic index...")
    vector_store.load(config.INDEX_PATH, config.METADATA_PATH)
    _print("  FAISS index loaded.")

    chunks = vector_store.get_chunks()
    if keyword_search.index_exists(config.BM25_INDEX_PATH):
        _print("[INDEX] Loading existing BM25 keyword index...")
        keyword_search.load(config.BM25_INDEX_PATH, chunks=chunks)
        _print("  BM25 index loaded.")
    else:
        _print("[INDEX] Building and caching BM25 keyword index from chunk store...")
        keyword_search.create_index(chunks)
        keyword_search.save(config.BM25_INDEX_PATH)
        _print(f"  BM25 index built and saved to {config.BM25_INDEX_PATH}")

    return True


# ── retrieval & search ────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    k: int = None,
    mode: str = None,
    alpha: float = None,
) -> list:
    """Search documents using hybrid, semantic, or keyword retrieval.

    Args:
        query: User question / search term.
        k: Number of top chunks to return.
        mode: "hybrid", "semantic", or "keyword".
        alpha: Weight for semantic search vs keyword search in hybrid mode (0.0 to 1.0).

    Returns:
        List of chunk dicts ranked by relevance.
    """
    k = k or config.TOP_K
    mode = (mode or config.DEFAULT_SEARCH_MODE).lower()
    alpha = config.HYBRID_ALPHA if alpha is None else float(alpha)

    # Ensure indices are loaded if available on disk
    if getattr(vector_store, "_index", None) is None:
        if not load_index_if_exists():
            raise RuntimeError("Indices not found on disk. Run python rag_pipeline.py --rebuild first.")

    if mode == "semantic":
        query_emb = emb_mod.embed_query(query)
        return vector_store.search(query_emb, k)

    if mode == "keyword":
        return keyword_search.search(query, k)

    if mode == "hybrid":
        pool_k = max(20, k * 3)
        query_emb = emb_mod.embed_query(query)
        semantic_results = vector_store.search(query_emb, pool_k)
        keyword_results = keyword_search.search(query, pool_k)

        # Reciprocal Rank Fusion (RRF)
        rrf_k = config.RRF_K
        rrf_scores = {}
        doc_data = {}

        for rank, res in enumerate(semantic_results, start=1):
            doc_id = res.get("chunk_id") or str(res.get("doc_idx"))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (alpha / (rrf_k + rank))
            doc_data[doc_id] = {
                "chunk_text": res["chunk_text"],
                "source_filename": res["source_filename"],
                "chunk_id": res["chunk_id"],
                "doc_idx": res.get("doc_idx"),
                "class_label": res.get("class_label", ""),
                "subject": res.get("subject", ""),
                "title": res.get("title", ""),
                "stream": res.get("stream", ""),
                "semantic_rank": rank,
                "semantic_score": res["score"],
                "keyword_rank": None,
                "keyword_score": None,
            }

        for rank, res in enumerate(keyword_results, start=1):
            doc_id = res.get("chunk_id") or str(res.get("doc_idx"))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + ((1.0 - alpha) / (rrf_k + rank))
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "chunk_text": res["chunk_text"],
                    "source_filename": res["source_filename"],
                    "chunk_id": res["chunk_id"],
                    "doc_idx": res.get("doc_idx"),
                    "class_label": res.get("class_label", ""),
                    "subject": res.get("subject", ""),
                    "title": res.get("title", ""),
                    "stream": res.get("stream", ""),
                    "semantic_rank": None,
                    "semantic_score": None,
                    "keyword_rank": rank,
                    "keyword_score": res["score"],
                }
            else:
                doc_data[doc_id]["keyword_rank"] = rank
                doc_data[doc_id]["keyword_score"] = res["score"]

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        fused_results = []
        for doc_id, rrf_sc in sorted_docs:
            item = dict(doc_data[doc_id])
            item["score"] = float(rrf_sc)
            item["search_type"] = "hybrid"
            fused_results.append(item)

        return fused_results

    raise ValueError(f"Unknown search mode '{mode}'. Expected 'hybrid', 'semantic', or 'keyword'.")


# ── query ─────────────────────────────────────────────────────────────────────

def rag_query(
    query: str,
    k: int = None,
    mode: str = None,
    alpha: float = None,
) -> dict:
    """Answer *query* and return a dict with ``answer``, ``sources``, ``query``."""
    k = k or config.TOP_K
    search_mode = mode or config.DEFAULT_SEARCH_MODE
    retrieved = hybrid_search(query, k=k, mode=search_mode, alpha=alpha)
    prompt = prompt_template.build_prompt(query, retrieved)
    answer = generate.generate_answer(prompt)
    return {
        "answer": answer,
        "sources": retrieved,
        "query": query,
        "search_mode": search_mode,
    }


def list_available_models():
    """Query and print all models currently available in LM Studio."""
    import urllib.request
    import json
    url = f"{config.LM_STUDIO_BASE_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
            _print("\n[LM STUDIO] Models currently available:")
            for m in models:
                _print(f"  * {m}")
            _print("")
    except Exception as e:
        _print(f"\n[LM STUDIO] Could not query models list: {e}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local RAG Pipeline with LM Studio (Hybrid Search)")
    parser.add_argument("--llm", type=str, help="Override LLM model name on the fly")
    parser.add_argument("--embed", type=str, help="Override Embedding model name on the fly")
    parser.add_argument("--list-models", action="store_true", help="List available models in LM Studio")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding FAISS and BM25 indices")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "keyword"],
        default=config.DEFAULT_SEARCH_MODE,
        help="Retrieval mode: hybrid / semantic / keyword (default: hybrid)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config.HYBRID_ALPHA,
        help="Semantic weight in hybrid mode (0.0=keyword only, 1.0=semantic only, default: 0.5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of parallel embedding workers (default: from config/auto)",
    )
    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    if args.llm:
        config.LLM_MODEL_NAME = args.llm
        _print(f"[CONFIG] Overriding LLM to: {config.LLM_MODEL_NAME}")

    if args.embed:
        config.EMBEDDING_MODEL_NAME = args.embed
        _print(f"[CONFIG] Overriding Embedding model to: {config.EMBEDDING_MODEL_NAME}")

    if args.workers:
        config.EMBED_WORKERS = args.workers
        _print(f"[CONFIG] Overriding embedding workers to: {config.EMBED_WORKERS}")

    # 1. Check that LM Studio models are reachable.
    try:
        verify_models()
    except RuntimeError as err:
        _print(f"\n[ERROR] {err}")
        sys.exit(1)

    _print("\n[INFO] Both models are reachable.")

    # 2. Rebuild if requested or if no index exists.
    if args.rebuild:
        _print("\n[INFO] --rebuild specified: Building indices from scratch...")
        build_index(reuse_embeddings=False)
    elif not load_index_if_exists():
        _print("\n[INFO] No saved index found -- building from scratch...")
        try:
            build_index(reuse_embeddings=True)
        except RuntimeError as err:
            _print(f"\n[ERROR] {err}")
            sys.exit(1)

    # 3. Interactive query loop.
    _print(f'\n[READY] Using LLM: {config.LLM_MODEL_NAME}')
    _print(f'[READY] Retrieval Mode: {args.mode.upper()}' + (f' (alpha={args.alpha})' if args.mode == 'hybrid' else ''))
    _print('[READY] Type your question (or "exit" to quit).\n')
    while True:
        try:
            user_input = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nGoodbye!")
            break
        if user_input.lower() in {"exit", "quit", ""}:
            _print("Goodbye!")
            break

        _print("\n[THINKING] ...")
        result = rag_query(user_input, mode=args.mode, alpha=args.alpha)
        _print("\n--- Answer ---\n")
        _print(result["answer"])
        _print("\n--- Sources ---")
        for src in result["sources"]:
            snippet = src["chunk_text"][:200].replace("\n", " ")
            details = []
            if src.get("semantic_rank") is not None:
                details.append(f"Sem #{src['semantic_rank']}")
            if src.get("keyword_rank") is not None:
                details.append(f"KW #{src['keyword_rank']}")
            detail_str = f" [{', '.join(details)}]" if details else ""
            label = src.get("class_label", "")
            subj = src.get("subject", "")
            book_tag = f" | {label} {subj}" if label else ""
            _print(f"  [{src['source_filename']}]{book_tag} (Score: {src['score']:.4f}{detail_str})")
            _print(f"    {snippet}...")
        _print("")


if __name__ == "__main__":
    main()
