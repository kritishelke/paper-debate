from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.pipeline.debate import create_debate_session
from backend.pipeline.memory import STORE
from training.prepare_po_training_data import prepare_training_data
from training.ratings_db import RatingsDB


app = FastAPI(title="PaperDebate", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DebateRequest(BaseModel):
    pdf_bytes: str | None = None
    abstract: str | None = None
    focus: str = ""


class RatingRequest(BaseModel):
    rater_id: str
    debate_quality: int = Field(ge=1, le=5)
    argument_novelty: int = Field(ge=1, le=5)
    claim_coverage: int = Field(ge=1, le=5)
    consensus_quality: int = Field(ge=1, le=5)
    notes: str = ""


@app.post("/debate")
async def post_debate(request: DebateRequest) -> dict[str, str]:
    session_id = await create_debate_session(pdf_bytes=request.pdf_bytes, abstract=request.abstract, focus=request.focus)
    return {"session_id": session_id}


@app.get("/debate/{session_id}/stream")
async def stream_debate(session_id: str) -> StreamingResponse:
    if session_id not in STORE.records:
        raise HTTPException(status_code=404, detail="Debate not found")

    async def event_source():
        async for event in STORE.stream(session_id):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/debate/{session_id}/graph")
async def get_graph(session_id: str) -> dict[str, Any]:
    return STORE.get(session_id).graph


@app.get("/debate/{session_id}/metrics")
async def get_metrics(session_id: str) -> dict[str, Any]:
    return STORE.get(session_id).metrics.to_dict()


@app.post("/debate/{session_id}/rating")
async def post_rating(session_id: str, request: RatingRequest) -> dict[str, str]:
    record = STORE.get(session_id)
    db = RatingsDB()
    db.upsert_debate(record)
    db.add_rating(session_id, request.model_dump())
    record.metrics.human_ratings.append(request.model_dump())
    if db.rating_count() in {10, 30, 100}:
        prepare_training_data(db_path=db.path)
    return {"status": "ok"}


@app.get("/training/status")
async def training_status() -> dict[str, Any]:
    db = RatingsDB()
    status = db.training_status()
    status["po_model_id"] = _current_po_model_id()
    status["po_model_version"] = status["po_model_id"] or "not-trained"
    return status


@app.post("/training/trigger")
async def training_trigger() -> dict[str, str]:
    db = RatingsDB()
    output = prepare_training_data(db_path=db.path)
    try:
        from training.fine_tune_po_model import start_fine_tune

        job_id = start_fine_tune(str(output))
    except Exception as exc:
        job_id = f"training-data-created:{exc}"
    return {"job_id": job_id}


def _current_po_model_id() -> str | None:
    import os

    return os.getenv("PO_MODEL_ID")
