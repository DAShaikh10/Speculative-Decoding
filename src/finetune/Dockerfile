# Fine-tuning image. Like the evaluation image, only the code is baked in: entrypoint.sh syncs the Python environment,
# here including the `finetune` dependency group, into the PVC on first start.
# Build from the repository root: docker build --platform linux/amd64 -f src/finetune/Dockerfile -t <image> .
FROM python:3.12-slim

# Triton (used by torch.compile) builds a small CUDA helper with a C compiler at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY src/finetune/entrypoint.sh pyproject.toml uv.lock ./
COPY src/speculative_decoding/ ./speculative_decoding/
COPY src/finetune/ ./finetune/
RUN chmod +x entrypoint.sh

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
