from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from web_app.runtime import EvaluationRuntime, RuntimeInitializationError

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("btp.web")
WEB_DIR = Path(__file__).resolve().parent
runtime = EvaluationRuntime()
request_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(runtime.initialize)
    app.state.runtime = runtime
    yield


app = FastAPI(title="BTP-2026 Five-Model RAG Evaluation", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


class AskRequest(BaseModel):
    domain: Literal["Legal", "ECE"]
    question: str = Field(min_length=1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def get_health():
    return runtime.health()


@app.post("/api/ask")
async def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")
    if runtime.status != "ready":
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Evaluation runtime is not ready",
                "runtime": runtime.health(),
            },
        )
    try:
        async with request_lock:
            return await run_in_threadpool(runtime.ask, payload.domain, question)
    except RuntimeInitializationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app.main:app", host="127.0.0.1", port=8000, reload=False)
