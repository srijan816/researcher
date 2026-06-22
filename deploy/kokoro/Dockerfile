# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KOKORO_MODEL_DIR=/data/kokoro-models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install \
      "fastapi>=0.115,<0.117" \
      "uvicorn[standard]>=0.30,<0.34" \
      "numpy>=1.26,<3" \
      "kokoro-onnx>=0.5.0,<0.6"

COPY deploy/kokoro/server.py /app/server.py

VOLUME ["/data/kokoro-models"]

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
