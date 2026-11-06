"""
Modul Tools — fungsi-fungsi "aksi" yang dipanggil router ketika intent
BUKAN pertanyaan informasi/kebijakan, melainkan permintaan spesifik
yang butuh pengecekan data langsung (bukan retrieval semantik).

Pola ini mengikuti konsep Tool Integration di Sesi 12.3.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MOCK_TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_tickets.csv"

_tickets_df: Optional[pd.DataFrame] = None
_lock = threading.Lock()


def _load_tickets() -> pd.DataFrame:
    """Memuat mock_tickets.csv sekali saja (semua kolom dibaca sebagai string)."""
    global _tickets_df
    if _tickets_df is None:
        with _lock:
            if _tickets_df is None:
                _tickets_df = pd.read_csv(MOCK_TICKETS_PATH, dtype=str).fillna("")
    return _tickets_df


def normalize_ticket_id(ticket_id: Optional[str]) -> str:
    """
    Menormalkan input menjadi format 'TICKET-XXXX'.
    'ticket-42', 'TICKET 42', ' 42 ' semuanya menjadi 'TICKET-0042'.
    Input yang tidak dikenali dikembalikan apa adanya (huruf besar, tanpa spasi tepi).
    """
    raw = (ticket_id or "").strip().upper()
    match = re.fullmatch(r"(?:TICKET[\s_\-#]*)?(\d{1,6})", raw)
    if match:
        return f"TICKET-{int(match.group(1)):04d}"
    return raw


def lookup_ticket(ticket_id: Optional[str]) -> Optional[Dict[str, str]]:
    """Mengembalikan data tiket sebagai dict, atau None jika tidak ditemukan / data tidak bisa dibaca."""
    normalized = normalize_ticket_id(ticket_id)
    if not normalized:
        return None
    try:
        df = _load_tickets()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.error("Gagal memuat %s: %s", MOCK_TICKETS_PATH, exc)
        return None
    rows = df[df["ticket_id"].str.strip().str.upper() == normalized]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def get_ticket_status(ticket_id: Optional[str]) -> str:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.

    Selalu mengembalikan string yang informatif; tiket yang tidak ditemukan
    atau data yang bermasalah tidak membuat aplikasi crash.
    """
    normalized = normalize_ticket_id(ticket_id)
    if not normalized:
        return "ID tiket tidak diberikan. Sertakan ID dengan format TICKET-XXXX, contoh: TICKET-0005."

    try:
        _load_tickets()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return "Data tiket sedang tidak dapat diakses. Silakan coba lagi nanti atau hubungi tim IT Support."

    ticket = lookup_ticket(normalized)
    if ticket is None:
        return (f"Tiket {normalized} tidak ditemukan. "
                "Periksa kembali ID tiket Anda (format TICKET-XXXX) atau hubungi tim IT Support.")

    return (f"Tiket {ticket['ticket_id']} — \"{ticket['subject']}\" "
            f"(kategori: {ticket['category']}). "
            f"Status: {ticket['status']}. "
            f"Update terakhir: {ticket['last_update']}.")
