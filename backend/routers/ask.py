"""POST /ask — main chat endpoint.

Pipeline:
  1. Resolve ticker from the question (src.orchestrator.ticker_resolver)
  2. If resolved → run 4 specialist agents (src.agents.Orchestrator)
  3. Always → multilingual FAISS RAG over the news corpus
  4. Build a context block + system prompt
  5. LLM (or template fallback) → return natural-language reply
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.models import AskRequest, AskResponse
from backend.db import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    orch = get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503,
                            detail="Orchestrator not available")
    try:
        result = orch.chat(req.question)
        return AskResponse(
            reply=result.reply,
            ticker=result.ticker,
            tools_called=list(result.tools_called or []),
            mode=result.mode,
            tokens=int(getattr(result, "tokens", 0) or 0),
        )
    except Exception as e:
        logger.exception("/ask failed")
        raise HTTPException(status_code=500, detail=str(e))
