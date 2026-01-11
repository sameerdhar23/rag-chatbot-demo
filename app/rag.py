# app/rag.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ---------- Paths (repo-safe) ----------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DEFAULT_INDEX_NAME = os.getenv("INDEX_NAME", "sr1107.index")
DEFAULT_DOCS_NAME = os.getenv("DOCS_NAME", "sr1107_docs.pkl")

INDEX_PATH = ARTIFACTS_DIR / DEFAULT_INDEX_NAME
DOCS_PATH = ARTIFACTS_DIR / DEFAULT_DOCS_NAME

# ---------- Tuning knobs ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

TOP_K = int(os.getenv("TOP_K", "8"))
MAX_CTX = int(os.getenv("MAX_CTX", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.55"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "4000"))
MAX_GEN_LEN = int(os.getenv("MAX_GEN_LEN", "256"))

# ---------- Singletons (load once) ----------
_index: Optional[faiss.Index] = None
_docs: Optional[Any] = None
_embedder: Optional[SentenceTransformer] = None
_generator: Optional[Any] = None


def _load_artifacts() -> Tuple[faiss.Index, Any]:
    global _index, _docs

    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))

    if _docs is None:
        if not DOCS_PATH.exists():
            raise FileNotFoundError(f"Docs pickle not found at: {DOCS_PATH}")
        with open(DOCS_PATH, "rb") as f:
            _docs = pickle.load(f)

    return _index, _docs


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_generator():
    global _generator
    if _generator is None:
        # Deterministic-ish generation for a grounded RAG demo
        _generator = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_length=MAX_GEN_LEN,
        )
    return _generator


def _doc_at(docs: Any, idx: int) -> Dict[str, Any]:
    """
    Assumes docs.pkl is list-like aligned to FAISS ids.
    If yours is dict-like, we handle that too.
    """
    if isinstance(docs, list):
        return docs[idx]
    if isinstance(docs, dict):
        # common patterns:
        if idx in docs:
            return docs[idx]
        if str(idx) in docs:
            return docs[str(idx)]
    raise TypeError("Unsupported docs.pkl structure. Expected list or dict keyed by ids.")


def _extract_text(d: Dict[str, Any]) -> str:
    for k in ("text", "chunk", "content", "page_content"):
        if k in d and isinstance(d[k], str):
            return d[k]
    # fallback: stringify
    return str(d)


def _extract_page(d: Dict[str, Any]) -> Optional[int]:
    for k in ("page", "page_num", "pageno"):
        if k in d:
            try:
                return int(d[k])
            except Exception:
                return None
    return None


def _extract_chunk_id(d: Dict[str, Any]) -> str:
    for k in ("chunk_id", "id", "chunkId"):
        if k in d and isinstance(d[k], str):
            return d[k]
    return ""


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    index, docs = _load_artifacts()
    embedder = _get_embedder()

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        d = _doc_at(docs, int(idx))
        hits.append(
            {
                "faiss_id": int(idx),
                "score": float(score),
                "page": _extract_page(d),
                "chunk_id": _extract_chunk_id(d),
                "text": _extract_text(d),
            }
        )
    return hits


def select_contexts(hits: List[Dict[str, Any]], max_ctx: int = MAX_CTX, min_score: float = MIN_SCORE) -> List[Dict[str, Any]]:
    strong = [h for h in hits if h["score"] >= min_score]
    return strong[:max_ctx]


def _build_context_block(contexts: List[Dict[str, Any]], max_chars: int = MAX_PROMPT_CHARS) -> str:
    parts: List[str] = []
    used = 0
    for c in contexts:
        header_bits = []
        if c.get("page") is not None:
            header_bits.append(f"Page {c['page']}")
        if c.get("chunk_id"):
            header_bits.append(c["chunk_id"])
        header = " | ".join(header_bits) if header_bits else f"Chunk {c.get('faiss_id','')}"
        block = f"[{header}]\n{(c.get('text') or '').strip()}\n\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip()


def _make_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx = _build_context_block(contexts)
    return (
        "You are an assistant answering questions using ONLY the provided context.\n"
        "If the answer is not in the context, say: \"Not found in the document.\"\n"
        "When possible, cite page numbers.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        "Answer:"
    )


def answer_question(question: str) -> Dict[str, Any]:
    hits = retrieve(question, top_k=TOP_K)
    contexts = select_contexts(hits, max_ctx=MAX_CTX, min_score=MIN_SCORE)

    if not contexts:
        return {
            "answer": "Not found in the document.",
            "contexts": [],
            "hits": hits[: min(5, len(hits))],
        }

    prompt = _make_prompt(question, contexts)
    gen = _get_generator()
    out = gen(prompt)[0].get("generated_text", "").strip()

    return {
        "answer": out,
        "contexts": contexts,
        "hits": hits[: min(5, len(hits))],
    }

