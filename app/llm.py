"""
Wrapper tipis untuk Gemini AI API.

Seluruh pemanggilan LLM (generate teks) dan embedding melewati modul ini agar:
  - API key dibaca dari environment variable (tidak pernah di-hardcode),
  - ada satu tempat untuk retry sederhana dan penerjemahan error ke LLMError,
  - modul lain (rag.py, router.py) tidak perlu tahu detail SDK.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional, TypeVar

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODELS = "gemini-3.5-flash-lite,gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 768

MAX_ATTEMPTS = 3
EMBED_BATCH_SIZE = 50

T = TypeVar("T")


class LLMError(Exception):
    """LLM / embedding gagal merespons (key hilang, kuota habis, jaringan, dll)."""

    def __init__(self, message: str, intent: str = "unknown"):
        super().__init__(message)
        self.intent = intent


_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def get_client() -> genai.Client:
    """Membuat client Gemini secara lazy supaya aplikasi tetap bisa start tanpa key."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("GEMINI_API_KEY belum diset. Isi di file .env atau lewat environment variable.")
    with _client_lock:
        if _client is None:
            if use_vertex_ai():
                # Key Vertex AI (express mode) memakai endpoint Vertex, bukan Gemini Developer API.
                _client = genai.Client(vertexai=True, api_key=api_key)
            else:
                _client = genai.Client(api_key=api_key)
    return _client


def use_vertex_ai() -> bool:
    return os.environ.get("USE_VERTEX_AI", "").strip().lower() in ("1", "true", "yes")


def reset_client() -> None:
    """Dipakai pengujian untuk memaksa pembuatan ulang client."""
    global _client
    _client = None


def generation_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GENERATION_MODEL)


def generation_models() -> List[str]:
    """Model utama + model cadangan (GEMINI_FALLBACK_MODELS, dipisah koma), tanpa duplikat."""
    fallbacks = os.environ.get("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)
    ordered = [generation_model()] + [m.strip() for m in fallbacks.split(",") if m.strip()]
    return list(dict.fromkeys(ordered))


def embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_dim() -> int:
    return int(os.environ.get("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM))


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    # Gangguan jaringan sesaat (timeout, koneksi terputus) juga layak dicoba ulang.
    return isinstance(exc, (ConnectionError, TimeoutError))


def _with_retry(call: Callable[[], T], what: str) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK melempar beragam tipe error
            last_exc = exc
            if attempt < MAX_ATTEMPTS and _is_retryable(exc):
                wait = 2 ** (attempt - 1)
                logger.warning("%s gagal (percobaan %d/%d): %s. Coba lagi dalam %ds.",
                               what, attempt, MAX_ATTEMPTS, exc, wait)
                time.sleep(wait)
                continue
            break
    logger.error("%s gagal: %s", what, last_exc)
    raise LLMError(f"{what} gagal: {type(last_exc).__name__}: {last_exc}") from last_exc


def generation_temperature() -> Optional[float]:
    """
    Temperature opsional lewat GEMINI_TEMPERATURE. Jika kosong, dipakai nilai bawaan model
    (model Gemini 3 direkomendasikan Google memakai temperature bawaan).
    """
    raw = os.environ.get("GEMINI_TEMPERATURE", "").strip()
    return float(raw) if raw else None


def generate_text(prompt: str) -> str:
    """Memanggil model Gemini dan mengembalikan teks jawaban (sinkron)."""
    client = get_client()
    last_error: Optional[LLMError] = None

    # Coba model utama; jika tetap gagal (mis. 503 lonjakan permintaan, 404 model dipensiunkan),
    # lanjut ke model cadangan.
    for model in generation_models():

        def _call(model=model) -> str:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=generation_temperature()),
            )
            return (response.text or "").strip()

        try:
            text = _with_retry(_call, f"Pemanggilan LLM ({model})")
        except LLMError as exc:
            logger.warning("Model %s gagal, mencoba model berikutnya jika ada.", model)
            last_error = exc
            continue
        if text:
            return text
        last_error = LLMError(f"LLM ({model}) mengembalikan respons kosong.")

    raise last_error or LLMError("Tidak ada model LLM yang dapat dipakai.")


def embed_texts(texts: List[str], task_type: str) -> np.ndarray:
    """
    Menghasilkan embedding ter-normalisasi L2 (float32) untuk daftar teks.

    task_type: 'RETRIEVAL_DOCUMENT' untuk dokumen knowledge base,
               'RETRIEVAL_QUERY' untuk pertanyaan user.
    """
    client = get_client()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        def _call(batch=batch):
            return client.models.embed_content(
                model=embedding_model(),
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=embedding_dim(),
                ),
            )

        result = _with_retry(_call, "Pemanggilan embedding")
        vectors.extend(list(e.values) for e in result.embeddings)

    if len(vectors) != len(texts):
        raise LLMError("Jumlah embedding tidak sama dengan jumlah teks input.")

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
