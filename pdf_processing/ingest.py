# ingest.py
"""Load raw source documents from disk and return clean plain-text strings.

Supported formats: .txt, .md, .pdf
PDFs are extracted in parallel using ProcessPoolExecutor.
Extracted text is written directly to disk (data/.cache_docs/<filename>.json) by workers
to avoid Windows IPC pipe buffer limitations (WinError 1450).
Subsequent runs load all documents in sub-second time from cache.
"""

import os
import re
import json
import pathlib
import logging
import warnings
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

# Suppress verbose font/width warnings from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pypdf")

try:
    from pypdf import PdfReader
    _pdf_import_error = None
except ImportError as e:  # pragma: no cover
    PdfReader = None
    _pdf_import_error = e

CACHE_DIR = os.path.join(config.BASE_DIR, "data", ".cache_docs")


# ── single-file loaders ───────────────────────────────────────────────────────

def _read_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def load_txt(path: str) -> str:
    return _read_file(path)


def load_markdown(path: str) -> str:
    return _read_file(path)


def load_pdf(path: str) -> str:
    """Extract text from all pages of a PDF."""
    if PdfReader is None:
        raise RuntimeError(
            f"pypdf is required for PDF ingestion but is not installed. "
            f"Run: pip install pypdf\nOriginal error: {_pdf_import_error}"
        )
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    return "\n".join(pages_text)


def clean_text(raw_text: str) -> str:
    """Collapse whitespace and strip boilerplate."""
    return re.sub(r"\s+", " ", raw_text).strip()


# ── worker function (must be module-level for pickling) ───────────────────────

def _load_single_file(args: tuple) -> dict:
    """Worker: load one file, save cleaned text directly to disk, and return small status dict.

    args = (full_path, doc_id, filename, ext, cache_file)
    Returns: {"ok": bool, "source_filename": str, "cache_file": str, "chars": int, "error": str}
    """
    full_path, doc_id, filename, ext, cache_file = args
    try:
        # Check cache first
        if cache_file and os.path.exists(cache_file):
            return {
                "ok": True,
                "source_filename": filename,
                "cache_file": cache_file,
                "cached": True,
            }

        if ext == ".txt":
            raw = load_txt(full_path)
        elif ext == ".md":
            raw = load_markdown(full_path)
        elif ext == ".pdf":
            raw = load_pdf(full_path)
        else:
            return {"ok": False, "source_filename": filename, "error": f"Unsupported ext {ext}"}

        cleaned = clean_text(raw)
        doc = {
            "doc_id": doc_id,
            "source_filename": filename,
            "text": cleaned,
        }

        # Save to cache file on disk directly from worker
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_cache = cache_file + f".tmp_{os.getpid()}"
            with open(temp_cache, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            os.replace(temp_cache, cache_file)

        # Return tiny status payload across IPC pipe (prevents WinError 1450)
        return {
            "ok": True,
            "source_filename": filename,
            "cache_file": cache_file,
            "chars": len(cleaned),
            "cached": False,
        }
    except Exception as e:
        return {"ok": False, "source_filename": filename, "error": str(e)}


# ── metadata loader ───────────────────────────────────────────────────────────

def _load_book_metadata(raw_docs_dir: str) -> Dict[str, Dict]:
    """Load metadata.json and return a filename -> metadata lookup dict."""
    search_paths = [
        pathlib.Path(raw_docs_dir) / "metadata.json",
        pathlib.Path(raw_docs_dir).parent / "metadata.json",
        pathlib.Path(config.BASE_DIR) / "data" / "metadata.json",
    ]
    for meta_path in search_paths:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                books = json.load(f)
            lookup = {b["filename"]: b for b in books}
            print(f"  [META] Loaded {len(lookup)} book entries from {meta_path}")
            return lookup

    print("  [META] metadata.json not found -- documents will not carry book metadata.")
    return {}


# ── main entry point ──────────────────────────────────────────────────────────

def load_all_documents(
    raw_docs_dir: str = config.RAW_DOCS_DIR,
    max_workers: int = None,
    use_cache: bool = True,
) -> List[Dict]:
    """Walk raw_docs_dir and load every supported file.

    PDFs are extracted in parallel. Text is saved to data/.cache_docs/
    so repeated runs load in milliseconds.

    Returns a list of dicts with keys:
        doc_id, source_filename, text
    and, for NCERT PDFs matched in metadata.json:
        class, class_label, subject, title, stream, total_pages
    """
    raw_path = pathlib.Path(raw_docs_dir)
    if not raw_path.is_dir():
        raise RuntimeError(
            f"RAW_DOCS_DIR does not exist: {raw_docs_dir}\n"
            f"Create it and place your .txt/.md/.pdf files inside."
        )

    book_meta = _load_book_metadata(raw_docs_dir)
    cache_dir = CACHE_DIR if use_cache else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Collect all files to process
    file_args = []
    for root, _, files in os.walk(raw_path):
        for filename in sorted(files):
            ext = pathlib.Path(filename).suffix.lower()
            if ext not in (".txt", ".md", ".pdf"):
                continue
            full_path = os.path.join(root, filename)
            doc_id = pathlib.Path(full_path).relative_to(raw_path).as_posix()
            cache_file = os.path.join(cache_dir, f"{filename}.json") if cache_dir else None
            file_args.append((full_path, doc_id, filename, ext, cache_file))

    total = len(file_args)
    if max_workers is None:
        # Use 6 workers on Windows to maintain responsive disk I/O without pipe limits
        max_workers = min(os.cpu_count() or 4, 6)

    # Separate already cached from uncached
    cached_indices = []
    to_extract = []

    for i, args in enumerate(file_args):
        cf = args[4]
        if cf and os.path.exists(cf):
            cached_indices.append(i)
        else:
            to_extract.append((i, args))

    cached_count = len(cached_indices)
    print(f"  [INGEST] {total} files total ({cached_count} cached, {len(to_extract)} to extract)...")

    if to_extract:
        print(f"  [INGEST] Extracting {len(to_extract)} files with {max_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(_load_single_file, args): i
                for i, args in to_extract
            }
            done = 0
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                result = future.result()
                done += 1
                if not result.get("ok"):
                    print(f"  [FAIL] {result['source_filename']}: {result.get('error')}")
                elif done % 5 == 0 or done == len(to_extract):
                    print(
                        f"  [INGEST] {done}/{len(to_extract)} files extracted"
                        f" ({done*100//len(to_extract)}%)",
                        flush=True
                    )

    # Now read all documents from cache files in order
    print(f"  [INGEST] Loading all {total} documents from disk cache...")
    documents: List[Dict] = []
    for full_path, doc_id, filename, ext, cache_file in file_args:
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception as e:
                print(f"  [WARN] Corrupt cache for {filename}, re-reading: {e}")
                raw = load_pdf(full_path) if ext == ".pdf" else _read_file(full_path)
                doc = {"doc_id": doc_id, "source_filename": filename, "text": clean_text(raw)}
        else:
            raw = load_pdf(full_path) if ext == ".pdf" else _read_file(full_path)
            doc = {"doc_id": doc_id, "source_filename": filename, "text": clean_text(raw)}

        # Enrich with book-level metadata
        if filename in book_meta:
            meta = book_meta[filename]
            doc["class"]       = meta.get("class")
            doc["class_label"] = meta.get("class_label", "")
            doc["subject"]     = meta.get("subject", "")
            doc["title"]       = meta.get("title", "")
            doc["stream"]      = meta.get("stream", "")
            doc["total_pages"] = meta.get("total_pages")

        label = doc.get("class_label", "")
        subj  = doc.get("subject", "")
        tag   = f" [{label} - {subj}]" if label and subj else ""
        print(f"  [OK] {filename}{tag}  ({len(doc['text']):,} chars)")
        documents.append(doc)

    return documents


if __name__ == "__main__":  # pragma: no cover
    import time
    t = time.time()
    docs = load_all_documents()
    elapsed = time.time() - t
    print(f"\nTotal: {len(docs)} document(s) loaded in {elapsed:.1f}s")
    with_meta = sum(1 for d in docs if "class" in d)
    print(f"  {with_meta} document(s) have book-level metadata.")
