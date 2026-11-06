"""
Test offline (tanpa API key / tanpa internet): tools, ekstraksi ticket_id,
intent detection berbasis keyword, dan kontrak response /ask.

Jalankan:  pytest -q
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from app import router, tools
from app.llm import LLMError

os.environ["INTENT_MODE"] = "keyword"


# ---------- tools ----------

def test_ticket_found():
    out = tools.get_ticket_status("TICKET-0005")
    assert "Reset password VPN gagal terus" in out
    assert "Diproses" in out
    assert "2026-08-27" in out


def test_ticket_not_found_does_not_crash():
    out = tools.get_ticket_status("TICKET-0042")
    assert "tidak ditemukan" in out


@pytest.mark.parametrize("raw", [None, "", "   ", "bukan-id", "TICKET-", "'; DROP TABLE--"])
def test_ticket_bad_input_does_not_crash(raw):
    assert isinstance(tools.get_ticket_status(raw), str)


@pytest.mark.parametrize("raw,expected", [
    ("TICKET-0005", "TICKET-0005"), ("ticket-5", "TICKET-0005"),
    (" 5 ", "TICKET-0005"), ("Ticket 15", "TICKET-0015"),
])
def test_normalize_ticket_id(raw, expected):
    assert tools.normalize_ticket_id(raw) == expected


# ---------- extract_ticket_id ----------

@pytest.mark.parametrize("query,expected", [
    ("apa status tiket TICKET-0042 saya?", "TICKET-0042"),
    ("cek ticket-7 dong", "TICKET-0007"),
    ("status tiket nomor 12", "TICKET-0012"),
    ("tiket #3 sudah selesai?", "TICKET-0003"),
    ("berapa jatah cuti saya?", None),
    ("cek status tiket saya", None),
])
def test_extract_ticket_id(query, expected):
    assert router.extract_ticket_id(query) == expected


# ---------- intent (keyword baseline) ----------

@pytest.mark.parametrize("query,expected", [
    ("Berapa jatah cuti tahunan saya?", "faq"),
    ("Bagaimana cara reset password VPN?", "faq"),
    ("Apakah perusahaan menyediakan asuransi kesehatan?", "faq"),
    ("Apa status tiket TICKET-0005 saya?", "ticket_status"),
    ("cek TICKET-0042", "ticket_status"),
    ("tolong cek status tiket saya", "ticket_status"),
    ("Bagaimana cara mengajukan tiket IT?", "faq"),
    ("Bagaimana cara melihat status tiket?", "faq"),
    ("", "unknown"),
    ("   ", "unknown"),
])
def test_keyword_intent(query, expected):
    assert asyncio.run(router.detect_intent(query)) == expected


# ---------- API ----------

@pytest.fixture()
def client(monkeypatch):
    from app import main

    async def fake_rag(query):
        return f"[RAG] {query}"

    monkeypatch.setattr(router, "answer_from_knowledge_base", fake_rag)
    monkeypatch.setattr(main.rag, "build_index", lambda: 0)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ask_faq_routes_to_rag(client):
    body = client.post("/ask", json={"query": "Berapa jatah cuti tahunan?"}).json()
    assert body["status"] == "success" and body["intent"] == "faq"
    assert body["data"].startswith("[RAG]")


def test_ask_ticket_found(client):
    body = client.post("/ask", json={"query": "status tiket TICKET-0001?"}).json()
    assert body == {"status": "success", "intent": "ticket_status",
                    "data": tools.get_ticket_status("TICKET-0001")}


def test_ask_ticket_not_found(client):
    r = client.post("/ask", json={"query": "status tiket TICKET-0042?"})
    assert r.status_code == 200
    assert r.json()["status"] == "not_found" and r.json()["intent"] == "ticket_status"


def test_ask_ticket_without_id(client):
    body = client.post("/ask", json={"query": "cek status tiket saya"}).json()
    assert body["status"] == "error" and body["intent"] == "ticket_status"


def test_ask_whitespace_query_is_unknown(client):
    body = client.post("/ask", json={"query": "   "}).json()
    assert body["status"] == "error" and body["intent"] == "unknown"


def test_ask_invalid_payload(client):
    r = client.post("/ask", json={})
    assert r.status_code == 422 and r.json()["status"] == "error"


def test_ask_llm_failure_returns_503(client, monkeypatch):
    async def boom(query):
        raise LLMError("quota habis")

    monkeypatch.setattr(router, "answer_from_knowledge_base", boom)
    r = client.post("/ask", json={"query": "Berapa jatah cuti?"})
    assert r.status_code == 503
    assert r.json()["status"] == "error" and r.json()["intent"] == "faq"
