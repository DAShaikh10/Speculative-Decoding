FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY entrypoint.sh evaluate.py pyproject.toml uv.lock ./
COPY src/speculative_decoding/ ./speculative_decoding/
RUN chmod +x entrypoint.sh

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
