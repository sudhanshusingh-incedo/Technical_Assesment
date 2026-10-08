# syntax=docker/dockerfile:1
# One image, several roles (ingest job, API, UI) selected by the compose `command`.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ---------- build: dependencies + package + local ONNX models ----------
FROM base AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
# Bake the embedding / BM25 / reranker models into the image so containers start offline.
COPY configs ./configs
COPY scripts/download_models.py ./scripts/download_models.py
ENV KASSIST_HOME=/app EMBEDDINGS__CACHE_DIR=/app/models
RUN python scripts/download_models.py

# ---------- runtime ----------
FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app/models /app/models
COPY --chown=app:app configs ./configs
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data/corpus ./data/corpus
COPY --chown=app:app src/kassist/ui/streamlit_app.py ./ui/streamlit_app.py
RUN mkdir -p /app/data/.parse_cache /app/state && chown -R app:app /app/data /app/state
ENV PATH=/opt/venv/bin:$PATH \
    KASSIST_HOME=/app \
    EMBEDDINGS__CACHE_DIR=/app/models \
    HF_HUB_OFFLINE=1 \
    LOG_JSON=true
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "kassist.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
