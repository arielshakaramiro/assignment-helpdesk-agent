"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import rag
from app.llm import LLMError
from app.router import handle_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bangun index FAISS sekali saat startup. Bila gagal (mis. API key belum diisi), app tetap hidup."""
    try:
        total = rag.build_index()
        logger.info("Startup: knowledge base terindeks (%d dokumen).", total)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Index belum bisa dibangun saat startup (%s). Akan dicoba lagi saat request pertama.", exc)
    yield


app = FastAPI(title="IT Helpdesk Assistant API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Pertanyaan bebas dari karyawan")


class AskResponse(BaseModel):
    status: str
    intent: str
    data: str


def _error(status_code: int, intent: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code,
                        content={"status": "error", "intent": intent, "data": message})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return _error(422, "unknown", "Request tidak valid: field 'query' (string, 1-1000 karakter) wajib diisi.")


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {"status": "ok", "index_ready": rag.is_index_ready()}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    Penanganan error:
      - intent tidak dikenali      -> 200, status="error", intent="unknown"
      - ID tiket tidak ada di query -> 200, status="error", intent="ticket_status"
      - LLM/embedding gagal         -> 503, status="error"
      - error tak terduga           -> 500, status="error" (detail hanya di log server)
    """
    try:
        result = await handle_query(request.query)
    except LLMError as exc:
        logger.error("LLM gagal merespons: %s", exc)
        return _error(503, exc.intent,
                      "Layanan AI sedang tidak dapat merespons. Silakan coba lagi beberapa saat lagi.")
    except Exception:  # noqa: BLE001
        logger.exception("Error tak terduga saat memproses /ask")
        return _error(500, "unknown", "Terjadi kesalahan internal pada server.")
    return AskResponse(**result)
