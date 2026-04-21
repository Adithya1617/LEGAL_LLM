# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# ---------- Stage 2: model-fetcher ----------
FROM python:3.12-slim AS model-fetcher
ENV PIP_NO_CACHE_DIR=1
RUN pip install huggingface_hub==0.25.2
ARG HF_MODEL_REPO=YOUR_HF_USER/legal-llm-llama32-3b
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN_VALUE=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") && \
    HF_TOKEN="$HF_TOKEN_VALUE" python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${HF_MODEL_REPO}', local_dir='/models', local_dir_use_symlinks=False)"

# ---------- Stage 3: runtime ----------
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=model-fetcher /models /models
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY src /app/src
COPY configs /app/configs
RUN sed -i 's|path: models/merged|path: /models|' /app/configs/serve.yaml
EXPOSE 8000
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
