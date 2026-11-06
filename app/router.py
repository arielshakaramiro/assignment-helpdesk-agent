"""
Router — menentukan apakah query user butuh RAG (informasi/kebijakan)
atau tool call (aksi spesifik, misal cek status tiket).

Pola ini mengikuti workflow router yang sudah dipelajari di Sesi 12.4:
  user input -> intent detection -> jika butuh data -> retrieval (RAG)
                                  -> jika butuh aksi -> tool call

Strategi intent detection (INTENT_MODE di .env):
  - "hybrid" (default): aturan keyword/regex dulu (cepat, deterministik, gratis);
                        LLM hanya dipakai untuk kasus yang benar-benar ambigu.
  - "keyword"         : hanya aturan keyword/regex (baseline, tanpa LLM).
  - "llm"             : semua query diklasifikasikan oleh LLM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Dict, Optional

from app.llm import LLMError, generate_text
from app.rag import answer_from_knowledge_base
from app.tools import get_ticket_status, lookup_ticket, normalize_ticket_id

logger = logging.getLogger(__name__)

INTENT_FAQ = "faq"
INTENT_TICKET = "ticket_status"
INTENT_UNKNOWN = "unknown"
VALID_INTENTS = {INTENT_FAQ, INTENT_TICKET, INTENT_UNKNOWN}

# Contoh yang cocok: TICKET-0042, ticket 42, tiket #7, tiket nomor 0005
TICKET_ID_RE = re.compile(
    r"\b(?:ticket|tiket)\s*[-_#:]?\s*(?:no\.?|nomor)?\s*[-_#:]?\s*(\d{1,6})\b",
    re.IGNORECASE,
)
TICKET_WORD_RE = re.compile(r"\b(?:ticket|tiket)\b", re.IGNORECASE)
STATUS_WORD_RE = re.compile(
    r"\b(?:status|cek|check|progres|progress|perkembangan|update|sampai mana|"
    r"sudah selesai|udah selesai|sudah diproses|diproses|ditangani|dikerjakan)\b",
    re.IGNORECASE,
)
HOWTO_WORD_RE = re.compile(
    r"\b(?:cara|bagaimana|gimana|prosedur|langkah|apa itu|apa yang|kenapa|mengapa|"
    r"syarat|kebijakan|how to)\b",
    re.IGNORECASE,
)
PERSONAL_RE = re.compile(r"\b(?:saya|aku|ku|punya saya)\b", re.IGNORECASE)

INTENT_PROMPT = """Klasifikasikan pesan karyawan berikut ke dalam TEPAT SATU kategori:
- faq: pertanyaan informasi/kebijakan/prosedur umum (cuti, password, perangkat, gaji, aturan kerja, dll).
- ticket_status: permintaan untuk mengecek status/progres tiket IT tertentu milik user.
- unknown: bukan pertanyaan yang jelas, sapaan kosong, atau tidak bisa dipahami.

Jawab hanya dengan satu kata: faq, ticket_status, atau unknown.

Pesan: {query}"""


def _intent_mode() -> str:
    return os.environ.get("INTENT_MODE", "hybrid").strip().lower()


def detect_intent_by_keyword(query: str) -> Optional[str]:
    """
    Baseline berbasis aturan. Mengembalikan intent, atau None jika ambigu
    (diserahkan ke LLM pada mode hybrid).

    Kriteria:
      1. Ada pola ID tiket (TICKET-0042 / tiket 42)                      -> ticket_status
      2. Menyebut tiket + kata status (status/cek/progres/...) tanpa
         kata tanya prosedur (cara/bagaimana/...)                         -> ticket_status
      3. Menyebut tiket + kata tanya prosedur ("cara membuat tiket")      -> faq
      4. Menyebut tiket tanpa petunjuk jelas, atau "status ... saya"
         tanpa kata tiket                                                 -> None (ambigu)
      5. Selain itu                                                       -> faq
    """
    text = query.strip()
    if not text:
        return INTENT_UNKNOWN

    if TICKET_ID_RE.search(text):
        return INTENT_TICKET

    has_ticket_word = bool(TICKET_WORD_RE.search(text))
    has_status_word = bool(STATUS_WORD_RE.search(text))
    has_howto_word = bool(HOWTO_WORD_RE.search(text))

    if has_ticket_word:
        if has_howto_word:
            return INTENT_FAQ
        if has_status_word:
            return INTENT_TICKET
        return None

    if has_status_word and PERSONAL_RE.search(text) and re.search(r"\bstatus\b", text, re.IGNORECASE):
        return None

    return INTENT_FAQ


def _classify_with_llm(query: str) -> str:
    answer = generate_text(INTENT_PROMPT.format(query=query)).lower()
    for label in (INTENT_TICKET, INTENT_FAQ, INTENT_UNKNOWN):
        if label in answer:
            return label
    logger.warning("Label intent dari LLM tidak dikenali: %r", answer)
    return INTENT_UNKNOWN


async def detect_intent(query: str) -> str:
    """
    Menentukan intent dari query user. Mengembalikan salah satu:
      - "faq"           -> pertanyaan informasi/kebijakan, arahkan ke RAG
      - "ticket_status" -> permintaan cek status tiket, arahkan ke tool call
      - "unknown"       -> tidak dapat dikenali
    """
    mode = _intent_mode()

    if not query.strip():
        return INTENT_UNKNOWN

    if mode == "llm":
        return await asyncio.to_thread(_classify_with_llm, query)

    by_keyword = detect_intent_by_keyword(query)
    if by_keyword is not None:
        return by_keyword

    if mode == "keyword":
        return INTENT_FAQ

    # hybrid + ambigu -> minta LLM; jika LLM gagal, jatuh ke faq (RAG akan menjawab "tidak tahu" bila perlu)
    try:
        return await asyncio.to_thread(_classify_with_llm, query)
    except LLMError as exc:
        logger.warning("Klasifikasi LLM gagal, fallback ke 'faq': %s", exc)
        return INTENT_FAQ


def extract_ticket_id(query: str) -> Optional[str]:
    """
    Mengekstrak ticket_id dari teks query dan menormalkannya ke 'TICKET-XXXX'.
    Mengembalikan None jika tidak ada ID tiket di dalam query.
    """
    match = TICKET_ID_RE.search(query)
    if not match:
        return None
    return normalize_ticket_id(match.group(1))


async def handle_query(query: str) -> Dict[str, str]:
    """
    Fungsi utama yang dipanggil oleh endpoint /ask.
    Menjalankan intent detection, lalu merutekan ke RAG atau tool call yang sesuai.

    Nilai `status`:
      success   -> permintaan berhasil diproses
      not_found -> intent dikenali, tetapi tiket yang diminta tidak ada di data
      error     -> intent tidak dikenali / input tidak lengkap
    LLMError dibiarkan naik ke main.py (dengan atribut `intent`) untuk diubah menjadi HTTP 503.
    """
    intent = await detect_intent(query)
    logger.info("Query=%r -> intent=%s", query, intent)

    if intent == INTENT_TICKET:
        ticket_id = extract_ticket_id(query)
        if ticket_id is None:
            return {
                "status": "error",
                "intent": INTENT_TICKET,
                "data": "Saya mendeteksi Anda ingin mengecek status tiket, tetapi ID tiket tidak ditemukan. "
                        "Sertakan ID dengan format TICKET-XXXX, contoh: TICKET-0005.",
            }
        found = lookup_ticket(ticket_id) is not None
        return {
            "status": "success" if found else "not_found",
            "intent": INTENT_TICKET,
            "data": get_ticket_status(ticket_id),
        }

    if intent == INTENT_FAQ:
        try:
            data = await answer_from_knowledge_base(query)
        except LLMError as exc:
            exc.intent = INTENT_FAQ
            raise
        return {"status": "success", "intent": INTENT_FAQ, "data": data}

    return {
        "status": "error",
        "intent": INTENT_UNKNOWN,
        "data": "Maaf, saya tidak memahami permintaan Anda. "
                "Silakan ajukan pertanyaan seputar kebijakan/IT, atau sertakan ID tiket (TICKET-XXXX) untuk cek status.",
    }
