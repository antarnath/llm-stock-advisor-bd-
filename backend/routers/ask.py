"""POST /ask — wraps Phase 10's Orchestrator.answer() and logs each
exchange to chat_log.sqlite.

The Orchestrator is a module-level singleton; NewsStore and the
prediction CSV open fresh per request, which is cheap enough for v1
single-user traffic.
"""
import time

from fastapi import APIRouter, HTTPException

from backend.db import log_chat
from backend.models import AskRequest, AskResponse
from src.orchestrator.orchestrator import Orchestrator

router = APIRouter()

# Shared singleton — orchestrator is stateless. Reusing across requests
# means we don't re-create the HTTP-client dispatch on every call.
_orch = Orchestrator()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer one user question.

    Latency is wall-clock for THIS endpoint, including the orchestrator's
    internal reads and (when configured) the external LLM call.
    """
    if not req.question.strip():
        # Pydantic's min_length=1 should catch this, but be defensive.
        raise HTTPException(400, "empty question")

    t0 = time.perf_counter()
    try:
        reply = _orch.answer(req.question)
    except Exception as exc:                                # noqa: BLE001
        # Orchestrator.answer() already swallows most errors; this guard
        # exists for truly unexpected failures (disk full, OOM, etc.).
        raise HTTPException(502, f"orchestrator failed: {exc}")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log_chat(
        question=req.question, reply=reply,
        session_id=req.session_id, latency_ms=latency_ms,
    )
    return AskResponse(
        reply=reply, session_id=req.session_id, latency_ms=latency_ms,
    )