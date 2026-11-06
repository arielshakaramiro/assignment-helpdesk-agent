"""
Menjalankan skenario uji end-to-end terhadap endpoint /ask yang sedang berjalan
(lokal maupun di dalam container Docker) dan menulis hasilnya ke tests/results.md.

Pemakaian:
    python tests/run_scenarios.py                       # default http://localhost:8000
    python tests/run_scenarios.py http://localhost:8000
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

# (nama, query, intent yang diharapkan, status yang diharapkan)
SCENARIOS = [
    ("1. FAQ - kebijakan cuti", "Berapa jatah cuti tahunan karyawan tetap?", "faq", "success"),
    ("2. FAQ - reset password VPN", "Bagaimana cara reset password VPN perusahaan?", "faq", "success"),
    ("3. Tiket ada", "Apa status tiket TICKET-0005 saya?", "ticket_status", "success"),
    ("4. Tiket tidak ada", "Tolong cek status tiket TICKET-0042", "ticket_status", "not_found"),
    ("5. Di luar cakupan", "Bagaimana prediksi cuaca di Jakarta besok?", "faq", "success"),
    ("6. Tambahan - minta cek tiket tanpa ID", "Tolong cek status tiket saya", "ticket_status", "error"),
    ("7. Tambahan - input kosong spasi", "   ", "unknown", "error"),
]


def post(query):
    req = urllib.request.Request(
        f"{BASE_URL}/ask",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main():
    lines = [f"# Hasil Pengujian /ask ({BASE_URL})", ""]
    passed = 0
    for name, query, exp_intent, exp_status in SCENARIOS:
        code, body = post(query)
        ok = body.get("intent") == exp_intent and body.get("status") == exp_status and code in (200,)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n  query : {query!r}\n  HTTP  : {code}\n  respon: {json.dumps(body, ensure_ascii=False)}\n")
        lines += [
            f"## {name} — {'PASS' if ok else 'FAIL'}",
            f"- Query: `{query}`",
            f"- Diharapkan: intent=`{exp_intent}`, status=`{exp_status}`",
            f"- HTTP {code}",
            "```json",
            json.dumps(body, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    summary = f"{passed}/{len(SCENARIOS)} skenario sesuai ekspektasi routing."
    print(summary)
    lines.append(f"**{summary}**")
    Path(__file__).with_name("results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
