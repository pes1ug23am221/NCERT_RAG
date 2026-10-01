# embeddings.py
# Purpose: wrap whichever embedding model is chosen behind a single stable interface.
# Supports parallel batch embedding with a live progress bar and automatic checkpointing.

import os
import json
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import config

CHECKPOINT_PATH = os.path.join(config.BASE_DIR, "data", ".embeddings_checkpoint.npz")


# ── VRAM-aware worker count ───────────────────────────────────────────────────

def _get_free_vram_mb() -> int:
    """Return free VRAM in MB via nvidia-smi, or 0 if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if lines:
            return int(lines[0])
    except Exception:
        pass
    return 0


def _recommended_workers(vram_free_mb: int, model_vram_mb: int = 500, max_workers: int = 8) -> int:
    """Recommend parallel workers based on free VRAM."""
    if vram_free_mb <= 0:
        return 2  # CPU fallback
    workers = max(1, vram_free_mb // model_vram_mb)
    return min(workers, max_workers)


# ── core API call ─────────────────────────────────────────────────────────────

def _call_embeddings_api_batch(texts: list[str], retries: int = 5) -> list[list[float]]:
    """Call LM Studio embeddings endpoint for a batch of texts. Retries on failure."""
    url = f"{config.LM_STUDIO_BASE_URL}/embeddings"
    payload = json.dumps({
        "model": config.EMBEDDING_MODEL_NAME,
        "input": texts
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return [item["embedding"] for item in data["data"]]
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            if attempt == retries:
                raise RuntimeError(f"Embedding batch failed after {retries} attempts: {e}") from e
            time.sleep(1.5 * attempt)
    return []


# ── progress bar ──────────────────────────────────────────────────────────────

class _ProgressBar:
    """Thread-safe terminal progress bar (ASCII-only, cp1252 safe)."""

    def __init__(self, total: int, initial: int = 0, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self._done = initial
        self._lock = threading.Lock()
        self._start = time.time()
        self._initial = initial
        self._render()

    def update(self, n: int = 1):
        with self._lock:
            self._done = min(self._done + n, self.total)
            self._render()

    def _render(self):
        pct = self._done / self.total if self.total else 1.0
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self._start
        recent_done = self._done - self._initial
        rate = recent_done / elapsed if elapsed > 0 else 0
        rem_batches = self.total - self._done
        eta = rem_batches / rate if rate > 0 else 0
        print(
            f"\r{self.prefix} [{bar}] {self._done}/{self.total}"
            f" ({pct*100:.1f}%) | {rate:.1f} b/s | ETA {eta:.0f}s   ",
            end="", flush=True
        )

    def finish(self):
        self._render()
        print()


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(path: str, embeddings: np.ndarray, done_mask: np.ndarray):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + f".tmp_{os.getpid()}"
        np.savez_compressed(temp_path, embeddings=embeddings, done_mask=done_mask)
        os.replace(temp_path, path)
    except Exception as e:
        print(f"\n  [WARN] Failed to save embedding checkpoint: {e}")


def _load_checkpoint(path: str, total_texts: int, dim: int):
    if not os.path.exists(path):
        return None, None
    try:
        data = np.load(path)
        embs = data["embeddings"]
        mask = data["done_mask"]
        if embs.shape == (total_texts, dim) and len(mask) == total_texts:
            return embs, mask
    except Exception:
        pass
    return None, None


# ── public API ────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    workers: int = None,
    checkpoint_interval: int = 25,
) -> np.ndarray:
    """Batch embed a list of texts using parallel requests with auto-resume.

    Args:
        texts:               List of strings to embed.
        batch_size:          Chunks per HTTP request (32 is optimal for nomic-embed).
        workers:             Parallel HTTP workers. Auto-detected from VRAM if None.
        checkpoint_interval: Save progress every N batches completed.

    Returns:
        (n, dim) float32 NumPy array.
    """
    if workers is None:
        workers = getattr(config, "EMBED_WORKERS", None)
    if workers is None:
        vram = _get_free_vram_mb()
        workers = _recommended_workers(vram)
        print(f"  [EMBED] Free VRAM: {vram} MB -> using {workers} parallel worker(s)")
    else:
        print(f"  [EMBED] Using {workers} parallel worker(s) (manual override)")

    total_texts = len(texts)
    dim = config.EMBEDDING_DIM

    # Check for existing checkpoint
    all_embeddings, done_mask = _load_checkpoint(CHECKPOINT_PATH, total_texts, dim)
    if all_embeddings is None:
        all_embeddings = np.zeros((total_texts, dim), dtype=np.float32)
        done_mask = np.zeros(total_texts, dtype=bool)

    # Prepare batches, skipping already completed ones
    batches_to_do = []
    total_batches = (total_texts + batch_size - 1) // batch_size
    already_done_batches = 0

    for b_idx in range(total_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, total_texts)
        if np.all(done_mask[start:end]):
            already_done_batches += 1
        else:
            batches_to_do.append((b_idx, start, end, texts[start:end]))

    if already_done_batches > 0:
        print(f"  [EMBED] Resumed from checkpoint: {already_done_batches}/{total_batches} batches already done")

    bar = _ProgressBar(total_batches, initial=already_done_batches, prefix="  [EMBED]")

    if batches_to_do:
        checkpoint_lock = threading.Lock()
        completed_since_save = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_info = {
                pool.submit(_call_embeddings_api_batch, batch_texts): (b_idx, start, end)
                for b_idx, start, end, batch_texts in batches_to_do
            }
            for future in as_completed(future_to_info):
                b_idx, start, end = future_to_info[future]
                embs = future.result()
                all_embeddings[start:end] = embs
                done_mask[start:end] = True
                bar.update(1)

                with checkpoint_lock:
                    completed_since_save += 1
                    if completed_since_save >= checkpoint_interval:
                        _save_checkpoint(CHECKPOINT_PATH, all_embeddings, done_mask)
                        completed_since_save = 0

    bar.finish()

    # Clean up checkpoint file on full completion
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except Exception:
            pass

    return all_embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returning a (1, dim) array."""
    emb = _call_embeddings_api_batch([query])[0]
    return np.array([emb], dtype=np.float32)


if __name__ == "__main__":
    vram = _get_free_vram_mb()
    w = _recommended_workers(vram)
    print(f"GPU free VRAM: {vram} MB -> recommended workers: {w}")
    vectors = embed_texts(["hello world", "test sentence", "NCERT physics"], workers=w)
    print(f"Shape: {vectors.shape}")
    print("Embeddings OK")
