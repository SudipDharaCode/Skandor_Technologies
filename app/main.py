"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.llm_client import GeminiClient, LLMError
from app.matcher import load_inventory
from app.pipeline import Analyzer
from app.schemas import AnalyzeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def build_analyzer() -> Analyzer:
    settings = get_settings()
    try:
        llm = GeminiClient(settings)
    except LLMError as exc:
        log.warning("LLM disabled (%s); /analyze will return degraded extractions", exc)
        llm = None
    return Analyzer(llm, load_inventory(settings.inventory_path), settings.max_image_bytes)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "analyzer"):  # tests inject their own analyzer
        app.state.analyzer = build_analyzer()
    yield


app = FastAPI(title="Brief to Priced Estimate", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    a: Analyzer = request.app.state.analyzer
    return {"status": "ok", "llm_enabled": a.llm is not None, "inventory_items": len(a.inventory)}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, brief: str = Form(""), image: UploadFile | None = File(None)):
    image_bytes = await image.read() if image is not None else None
    if not brief.strip() and not image_bytes:
        raise HTTPException(status_code=422, detail="provide a non-empty brief and/or an image")
    if len(brief) > 5000:
        raise HTTPException(status_code=422, detail="brief too long (max 5000 characters)")
    a: Analyzer = request.app.state.analyzer
    # The pipeline is synchronous (blocking HTTP + OpenCV); run it off the event loop.
    return await run_in_threadpool(a.analyze, brief, image_bytes or None, image.content_type if image else None)
