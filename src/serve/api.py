from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.common.schemas import ExtractRequest, ExtractResponse
from src.serve.extractor import Extractor, build_extractor

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading_extractor")
    _state["extractor"] = build_extractor(Path("configs/serve.yaml"))
    log.info("extractor_ready", version=_state["extractor"].model_version)
    yield
    _state.clear()


app = FastAPI(title="Legal LLM — Contract Clause Extractor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://demo:7860"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/version")
def version() -> dict:
    ex: Extractor | None = _state.get("extractor")
    return {"model": ex.model_version if ex else "not-loaded"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    ex: Extractor | None = _state.get("extractor")
    if ex is None:
        raise HTTPException(status_code=503, detail="extractor not loaded")
    log.info("extract_request", text_len=len(req.text))
    resp = ex.extract(req.text, req.clause_types)
    log.info(
        "extract_response",
        n_clauses=len(resp.clauses),
        latency_ms=resp.latency_ms,
    )
    return resp
