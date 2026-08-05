# syntax=docker/dockerfile:1
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/huggingface/sentence-transformers

WORKDIR /app

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --default-index "${PYTORCH_INDEX_URL}" torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app/data/runtime/uploads /cache/huggingface \
    && chown -R app:app /app /cache/huggingface

COPY --chown=app:app src ./src
COPY --chown=app:app data/knowledge.txt ./data/knowledge.txt

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"]

STOPSIGNAL SIGTERM

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
