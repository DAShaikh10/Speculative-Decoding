FROM python:3.12-slim

# Triton (used by torch.compile) builds a small CUDA helper with a C compiler at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY entrypoint.sh evaluate.py pyproject.toml uv.lock ./
COPY src/speculative_decoding/ ./speculative_decoding/
RUN chmod +x entrypoint.sh

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
