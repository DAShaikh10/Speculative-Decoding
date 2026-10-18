# Slim image: only the server code. Python packages (torch etc.) are installed on first
# container start into the PVC (/cache/pylibs) by entrypoint.sh and reused on restarts.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt entrypoint.sh server.py ./
RUN chmod +x entrypoint.sh

ENV HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["/app/entrypoint.sh"]
