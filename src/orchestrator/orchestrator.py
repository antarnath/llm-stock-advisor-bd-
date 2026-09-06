"""
LLM Orchestrator — the top-level entry point for the chat advisor.

Workflow:
  1. Resolve the DSE ticker from the user's question (if any).
  2. If a ticker is found, eagerly run all 3 specialist agents + RAG.
  3. Build a context prompt with the agent evidence + RAG citations.
  4. Call the LLM (or stub) to produce a natural-language answer.
  5. If the LLM is unavailable, fall back to a deterministic template
     that summarizes the evidence pack.

Public API:
  - answer(question: str) -> str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.agents import Orchestrator as AgentsOrchestrator, AdvisorQuery
from src.agents.orchestrator import STOCK_UNIVERSE
from src.llm import LLMClient, ChatMessage
from .ticker_resolver import resolve_ticker
from .tools import ToolRegistry, TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a financial advisor for the Bangladesh stock market.
You answer questions about 30 DSE-listed companies using evidence from
specialist agents (technical, news, risk) and a multilingual news index.

Rules:
1. ALWAYS cite the source of any specific number or claim
   (e.g. "per the technical agent" or "per RAG hit [1]").
2. NEVER invent prices, returns, or news — only use the evidence provided.
3. If the evidence is insufficient, say so explicitly.
4. When giving a recommendation (BUY / SELL / HOLD), state the confidence
   and the consensus among agents (unanimous / majority / split).
5. Use Bangla only if the user wrote in Bangla; otherwise respond in English.
"""


@dataclass
class ChatAnswer:
    """Wrapper around the orchestrator's reply."""
    reply: str
    ticker: Optional[str]
    tools_called: list[str]
    mode: str                          # "llm" | "stub" | "template"
    tokens: int = 0


class Orchestrator:
    """LLM-orchestrated financial advisor.

    Composes:
      - agents_orchestrator (src.agents.Orchestrator) — specialist agents
      - rag_retriever (src.rag.retriever.NewsRetriever) — multilingual news
      - llm_client (src.llm.LLMClient) — OpenAI-compatible LLM or stub
      - tools (src.orchestrator.tools.ToolRegistry) — tool dispatch

    The orchestrator is stateless — safe to use as a module-level singleton.
    """

    def __init__(self,
                 project_root: Optional[Path] = None,
                 llm_client: Optional[LLMClient] = None,
                 agents_orchestrator: Optional[AgentsOrchestrator] = None,
                 ):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.llm = llm_client or LLMClient()
        self.agents = agents_orchestrator or AgentsOrchestrator(
            models_dir=self.project_root / "models" / "baseline",
            data_dir=self.project_root / "data" / "processed",
            sentiment_csv=self.project_root / "results" / "sentiment" / "news_scored.csv",
            rag_index_dir=self.project_root / "models" / "rag",
        )

        # Lazy-init RAG
        self._rag = None
        try:
            from src.rag.retriever import NewsRetriever
            self._rag = NewsRetriever(
                index_dir=self.project_root / "models" / "rag",
            )
        except Exception as e:
            logger.warning("RAG retriever not loaded: %s", e)

        self.tools = ToolRegistry(
            agents_orchestrator=self.agents,
            rag_retriever=self._rag,
            project_root=self.project_root,
        )

    # ------------------------------------------------------------------
    def answer(self, question: str) -> str:
        """Answer a user question. Returns a natural-language reply string.

        Convenience wrapper used by the backend /ask router.
        """
        result = self.chat(question)
        return result.reply

    # ------------------------------------------------------------------
    def chat(self, question: str) -> ChatAnswer:
        """End-to-end chat: parse → evidence → LLM (or template)."""
        ticker = resolve_ticker(question, STOCK_UNIVERSE)

        # Always run evidence pack if a ticker is resolved
        evidence_pack = ""
        tools_called = []
        if ticker:
            tools_called.append("multi_agent_evidence")
            evidence_pack = self._build_evidence_pack(ticker, question)

        # Always run a generic RAG query if no ticker-specific hit
        rag_pack = ""
        if self._rag is not None:
            try:
                hits = self._rag.query(question, top_k=3)
                if hits:
                    rag_pack = "\n\n".join(
                        f"[{i+1}] {h.date} | {h.stock} | sim={h.similarity:.2f}\n"
                        f"    {h.headline}"
                        for i, h in enumerate(hits)
                    )
                    tools_called.append("rag_search")
            except Exception:
                pass

        # Try the LLM first
        if self.llm.is_live:
            try:
                return self._llm_answer(question, ticker, evidence_pack, rag_pack,
                                       tools_called)
            except Exception as e:
                logger.warning("LLM call failed: %s — falling back to template", e)

        # Fallback: deterministic template
        return self._template_answer(question, ticker, evidence_pack, rag_pack,
                                     tools_called)

    # ------------------------------------------------------------------
    def _build_evidence_pack(self, ticker: str, question: str) -> str:
        """Run the 3 specialist agents and assemble a textual pack."""
        q = AdvisorQuery(user_query=question, stock=ticker,
                         user_profile="moderate", capital_bdt=100_000.0)
        try:
            result = self.agents.advise(q)
        except Exception as e:
            return f"[evidence pack failed: {e}]"

        lines = [
            f"Ticker: {result.stock}",
            f"Recommendation: {result.recommendation}  "
            f"(confidence {result.confidence:.0%}, consensus {result.consensus})",
            f"Ensemble signal: {result.ensemble_signal:+.3f}",
            f"Position size: {result.position_pct*100:.1f}%  "
            f"(৳{result.position_bdt:,.0f})",
            "",
            "Reasoning chain:",
        ]
        for bullet in result.reasoning:
            lines.append(f"  - {bullet}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _llm_answer(self, question, ticker, evidence_pack, rag_pack,
                    tools_called) -> ChatAnswer:
        # Build context block
        ctx_parts = []
        if evidence_pack:
            ctx_parts.append("=== MULTI-AGENT EVIDENCE ===\n" + evidence_pack)
        if rag_pack:
            ctx_parts.append("=== RAG NEWS HITS ===\n" + rag_pack)
        context = "\n\n".join(ctx_parts) if ctx_parts else "(no specialist evidence needed)"

        user_msg = (
            f"User question: {question}\n\n"
            f"Resolved ticker: {ticker or '(none)'}\n\n"
            f"{context}"
        )

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_msg),
        ]
        resp = self.llm.chat(messages, temperature=0.2, max_tokens=1024)
        return ChatAnswer(
            reply=resp.content or "(empty LLM reply)",
            ticker=ticker,
            tools_called=tools_called,
            mode="llm",
            tokens=resp.total_tokens,
        )

    # ------------------------------------------------------------------
    def _template_answer(self, question, ticker, evidence_pack, rag_pack,
                         tools_called) -> ChatAnswer:
        """Deterministic template (no LLM)."""
        lines = []
        if ticker:
            lines.append(f"Analysis for **{ticker}**:\n")
            lines.append(evidence_pack or "(no evidence available)")
        else:
            lines.append("I couldn't identify a specific DSE stock in your question.")
            lines.append("")
            lines.append("You can ask about any of these 30 DSE-listed companies:")
            lines.append(", ".join(STOCK_UNIVERSE[:15]))
            lines.append("... (and 15 more)")

        if rag_pack:
            lines.append("\n\nRelevant news from the corpus:")
            lines.append(rag_pack)

        lines.append("\n\n_Reply generated via deterministic template "
                     "(no LLM configured)._")
        return ChatAnswer(
            reply="\n".join(lines),
            ticker=ticker,
            tools_called=tools_called,
            mode="template",
            tokens=0,
        )
