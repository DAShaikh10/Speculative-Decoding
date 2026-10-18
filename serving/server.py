"""Minimal HTTP inference server for a T5 model (default: t5-small).

Endpoints
  GET  /health    200 once the model is loaded (used as the k8s readiness probe)
  GET  /info      model name, device, dtype, parameter count
  POST /generate  run one generation, return text + token ids + latency

All settings come from environment variables so the same image works for any T5
checkpoint (t5-small, t5-base, ..., t5-3b), see k8s/server.yaml.tmpl.
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_NAME = os.environ.get("MODEL_NAME", "google-t5/t5-small")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "t5-small")
DTYPE = os.environ.get("DTYPE", "float32")  # float32 | float16 | bfloat16
PORT = int(os.environ.get("PORT", "8000"))

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Original T5 checkpoints are multi-task trained, the task is chosen by a text prefix.
TASK_PREFIXES = {
    "translate_en_de": "translate English to German: ",
    "summarize": "summarize: ",
    "raw": "",  # caller supplies the full prompt
}

state = {}
# One GPU, one request at a time: keeps latency numbers clean for benchmarking.
gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if DTYPE not in DTYPES:
        raise ValueError(f"DTYPE must be one of {list(DTYPES)}, got {DTYPE!r}")
    print(f"Loading {MODEL_NAME} as {DTYPE} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=DTYPES[DTYPE])
    model.to(device).eval()
    state.update(tokenizer=tokenizer, model=model, device=device)
    print("Model ready.", flush=True)
    yield
    state.clear()


app = FastAPI(title="T5 inference server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Input text (without task prefix unless task='raw').")
    task: Literal["translate_en_de", "summarize", "raw"] = "translate_en_de"
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.0, ge=0.0, description="0 = greedy (argmax), >0 = sampling.")
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    model: str
    text: str
    token_ids: list[int]
    num_new_tokens: int
    latency_ms: float


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok"}


@app.get("/info")
def info():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    model = state["model"]
    return {
        "model": SERVED_NAME,
        "checkpoint": MODEL_NAME,
        "device": state["device"],
        "dtype": DTYPE,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    tokenizer, model, device = state["tokenizer"], state["model"], state["device"]

    inputs = tokenizer(TASK_PREFIXES[req.task] + req.text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    gen_kwargs = {"max_new_tokens": req.max_new_tokens, "num_beams": 1}
    if req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature,
                          top_k=req.top_k or 0, top_p=req.top_p or 1.0)
    else:
        gen_kwargs.update(do_sample=False)

    with gpu_lock:
        if req.seed is not None:
            torch.manual_seed(req.seed)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

    # T5 is encoder-decoder: output starts with the decoder start token (pad), drop it.
    token_ids = out[0].tolist()[1:]
    return GenerateResponse(
        model=SERVED_NAME,
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        num_new_tokens=len(token_ids),
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
