"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS,
melakukan semantic search, lalu menghasilkan jawaban menggunakan LLM.

Alur ini serupa dengan yang sudah dibangun di Assignment 3 (Semantic Search &
LLM Application) dan Sesi 11 (Building a Simple RAG System) — di assignment ini,
alur tersebut dibungkus menjadi fungsi yang dipanggil oleh router.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from app.llm import LLMError, embed_texts, generate_text

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent / "data" / "it_helpdesk_knowledge_base.csv"

NO_ANSWER_MESSAGE = (
    "Maaf, saya tidak menemukan informasi tersebut di knowledge base. "
    "Silakan hubungi tim IT Support atau HR untuk bantuan lebih lanjut."
)

# Index dibangun sekali saat startup (lihat lifespan di main.py), bukan di setiap request.
_index: Optional[faiss.Index] = None
_documents: Optional[List[str]] = None
_df: Optional[pd.DataFrame] = None
_build_lock = threading.Lock()


def _top_k() -> int:
    return int(os.environ.get("TOP_K", "3"))


def _min_score() -> float:
    return float(os.environ.get("RAG_MIN_SCORE", "0"))


def is_index_ready() -> bool:
    return _index is not None


def _get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """Embedding satu teks, bentuk (1, dim), float32 dan ter-normalisasi."""
    return embed_texts([text], task_type)


def build_index() -> int:
    """
    Membangun index FAISS dari knowledge base CSV.

    Unit retrievable: satu baris FAQ (kategori + pertanyaan + jawaban) digabung
    menjadi satu dokumen, sehingga pencarian bisa cocok ke pertanyaan maupun isi jawaban.
    Vektor dinormalisasi L2 dan disimpan di IndexFlatIP, sehingga skor inner product
    setara cosine similarity (1.0 = identik).

    Mengembalikan jumlah dokumen yang terindeks.
    """
    global _index, _documents, _df
    with _build_lock:
        df = pd.read_csv(KNOWLEDGE_BASE_PATH, dtype=str).fillna("")
        documents = [
            f"Kategori: {row.category}\nPertanyaan: {row.question}\nJawaban: {row.answer}"
            for row in df.itertuples(index=False)
        ]
        vectors = embed_texts(documents, "RETRIEVAL_DOCUMENT")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        _df, _documents, _index = df, documents, index
        logger.info("Index FAISS siap: %d dokumen, dimensi %d.", index.ntotal, vectors.shape[1])
        return index.ntotal


def _ensure_index() -> None:
    """Fallback lazy-loading jika index belum terbentuk saat startup (misal key baru diisi)."""
    if _index is None:
        build_index()


def semantic_search(query: str, top_k: Optional[int] = None) -> List[Dict]:
    """
    Semantic search terhadap index FAISS.

    Mengembalikan daftar dict (urut skor tertinggi):
    {faq_id, category, question, answer, score}
    """
    _ensure_index()
    assert _index is not None and _df is not None

    k = min(top_k or _top_k(), _index.ntotal)
    query_vec = _get_embedding(query, "RETRIEVAL_QUERY")
    scores, ids = _index.search(query_vec, k)

    results: List[Dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = _df.iloc[int(idx)]
        results.append({
            "faq_id": row["faq_id"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "score": float(score),
        })
    return results


def build_prompt(query: str, results: List[Dict]) -> str:
    """Menyusun prompt terstruktur: Persona / Task / Context / Format."""
    context = "\n\n".join(
        f"[{i}] ({r['category']}) Q: {r['question']}\n    A: {r['answer']}"
        for i, r in enumerate(results, start=1)
    )
    return f"""## Persona
Kamu adalah IT Helpdesk Assistant internal perusahaan yang ramah, ringkas, dan akurat.

## Task
Jawab pertanyaan karyawan di bawah ini HANYA berdasarkan informasi pada bagian Context.
Jangan menambahkan fakta, angka, atau kebijakan yang tidak tertulis di Context.
Jika Context tidak memuat jawaban yang relevan, jawab persis dengan kalimat berikut dan tanpa tambahan:
"{NO_ANSWER_MESSAGE}"

## Context
{context}

## Format
- Gunakan bahasa Indonesia yang sopan.
- Jawab langsung dalam 1-3 kalimat, tanpa pembukaan atau penutup yang bertele-tele.
- Jangan menyebut kata "Context" atau nomor sumber di dalam jawaban.

## Pertanyaan
{query}

## Jawaban"""


async def answer_from_knowledge_base(query: str) -> str:
    """
    Fungsi yang dipanggil router.py ketika intent = 'faq'.
    Alur: semantic_search -> (opsional) cek skor minimum -> build prompt -> panggil LLM.
    Mengangkat LLMError jika embedding/LLM gagal merespons (ditangani di main.py).
    """
    results = await asyncio.to_thread(semantic_search, query)
    if not results:
        return NO_ANSWER_MESSAGE

    min_score = _min_score()
    if min_score > 0 and results[0]["score"] < min_score:
        logger.info("Skor retrieval tertinggi %.3f < %.3f, query dianggap di luar cakupan.",
                    results[0]["score"], min_score)
        return NO_ANSWER_MESSAGE

    prompt = build_prompt(query, results)
    return await asyncio.to_thread(generate_text, prompt)
