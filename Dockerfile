# Image dasar Python ringan
FROM python:3.11-slim

# Output log langsung tampil (tanpa buffering) dan tidak menulis file .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Salin requirements dulu agar layer instalasi dependency ter-cache
# selama requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install -r requirements.txt

# Salin source code dan data
COPY app/ ./app/
COPY data/ ./data/

# Jalankan sebagai user non-root
RUN useradd --create-home --uid 1000 appuser
USER appuser

# API key TIDAK dimasukkan ke image. Berikan saat runtime:
#   docker run --env-file .env ...
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# host 0.0.0.0 supaya bisa diakses dari luar container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
