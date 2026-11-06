"""
Diagnosa API key Gemini: mencoba generate + embedding lewat dua jalur dan
melaporkan mana yang berfungsi. Key tidak pernah dicetak.

Pemakaian (dari root project, venv aktif):
    python tests/check_key.py
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEN_MODELS = list(dict.fromkeys([os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")]
                               + os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash").split(",")))
EMB_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

if not KEY:
    sys.exit("GEMINI_API_KEY kosong di .env")

print(f"Key terbaca: panjang {len(KEY)}, prefix '{KEY[:3]}'\n")


def short(exc):
    return f"{type(exc).__name__}: {str(exc)[:160]}"


def try_path(name, client):
    print(f"== {name}")
    ok_all = True
    gen_ok = False
    for model in GEN_MODELS:
        try:
            r = client.models.generate_content(model=model, contents="Balas satu kata: OK")
            print(f"   generate ({model}): PASS -> {(r.text or '').strip()[:30]!r}")
            gen_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"   generate ({model}): FAIL -> {short(exc)}")
    ok_all = ok_all and gen_ok
    try:
        r = client.models.embed_content(model=EMB_MODEL, contents=["tes"])
        print(f"   embedding ({EMB_MODEL}): PASS -> dim {len(r.embeddings[0].values)}")
    except Exception as exc:  # noqa: BLE001
        ok_all = False
        print(f"   embedding ({EMB_MODEL}): FAIL -> {short(exc)}")
    print()
    return ok_all


a = try_path("Jalur A: Gemini Developer API (AI Studio)  -> USE_VERTEX_AI=false", genai.Client(api_key=KEY))
b = try_path("Jalur B: Vertex AI express mode            -> USE_VERTEX_AI=true", genai.Client(vertexai=True, api_key=KEY))

if a:
    print("KESIMPULAN: pakai Jalur A (default, tidak perlu ubah apa pun).")
elif b:
    print("KESIMPULAN: pakai Jalur B. Tambahkan USE_VERTEX_AI=true di .env.")
else:
    print("KESIMPULAN: key ditolak di kedua jalur. Lihat bagian Troubleshooting di README.")
