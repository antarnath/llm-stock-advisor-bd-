"""
Tool definitions — what the LLM orchestrator can call.

Each tool is a small wrapper around one of the specialist agents or the
RAG retriever. The LLM is given these descriptions in JSON-schema form;
when it calls one, the orchestrator dispatches and folds the result back
into the conversation.

Tools:
  - run_technical_agent(ticker)
  - run_news_agent(ticker, query)
  - run_risk_agent(ticker)
  - run_portfolio_agent(ticker, profile, capital_bdt, ensemble_signal)
  - rag_search(query, ticker, top_k)
  - get_latest_prediction(ticker)
  - get_recent_news(ticker, days)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# JSON-schema-ish tool descriptions for the LLM prompt
TOOL_DESCRIPTIONS = [
    {
        "name": "run_technical_agent",
        "description": "Run the technical analysis agent for a DSE ticker. "
                       "Loads the per-stock LSTM/XGBoost model and predicts "
                       "the next-day return. Use when the user asks about "
                       "price forecasts, technical signals, or trend.",
        "parameters": {
            "ticker": {"type": "string", "description": "DSE ticker code, e.g. 'BATBC'"},
        },
    },
    {
        "name": "run_news_agent",
        "description": "Run the news analyst agent for a DSE ticker. "
                       "Aggregates sentiment from recent news articles and "
                       "retrieves RAG citations. Use when the user asks "
                       "about news impact, sentiment, or recent events.",
        "parameters": {
            "ticker": {"type": "string", "description": "DSE ticker code"},
            "query":  {"type": "string", "description": "Original user question for RAG context"},
        },
    },
    {
        "name": "run_risk_agent",
        "description": "Run the risk analyst agent for a DSE ticker. "
                       "Computes volatility, Sharpe ratio, and max drawdown "
                       "from the last 252 trading days. Use when the user "
                       "asks about risk, volatility, or stability.",
        "parameters": {
            "ticker": {"type": "string", "description": "DSE ticker code"},
        },
    },
    {
        "name": "rag_search",
        "description": "Search the multilingual financial news index for "
                       "articles matching a natural-language query. Supports "
                       "English and Bangla. Use for open-ended questions "
                       "about Bangladesh stock market events.",
        "parameters": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "ticker": {"type": "string", "description": "Optional DSE ticker filter"},
            "top_k": {"type": "integer", "description": "Number of hits to return (default 5)"},
        },
    },
    {
        "name": "get_latest_prediction",
        "description": "Read the latest stored prediction for a ticker "
                       "(from results/multimodal/predictions_*.csv). "
                       "Use to show the user the model's most recent call.",
        "parameters": {
            "ticker": {"type": "string", "description": "DSE ticker code"},
        },
    },
]


class ToolRegistry:
    """Dispatches tool calls from the LLM to the underlying agents."""

    def __init__(self,
                 agents_orchestrator,
                 rag_retriever=None,
                 project_root=None):
        self.orch = agents_orchestrator
        self.rag = rag_retriever
        self.project_root = project_root

    def call(self, name: str, args: dict) -> str:
        """Invoke a tool by name with the given args dict.

        Returns a string result (suitable for folding back into the LLM
        context). On error, returns a short error message.
        """
        try:
            if name == "run_technical_agent":
                return self._technical(args)
            elif name == "run_news_agent":
                return self._news(args)
            elif name == "run_risk_agent":
                return self._risk(args)
            elif name == "rag_search":
                return self._rag(args)
            elif name == "get_latest_prediction":
                return self._prediction(args)
            else:
                return f"[unknown tool: {name}]"
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"[tool {name} error: {e}]"

    # ------------------------------------------------------------------
    def _technical(self, args):
        ev = self.orch.technical_agent.analyze(args["ticker"])
        return (
            f"signal={ev.signal:+.3f}  confidence={ev.confidence:.2f}\n"
            f"headline: {ev.headline}\n"
            f"details: {ev.details}"
        )

    def _news(self, args):
        ev = self.orch.news_agent.analyze(
            args["ticker"], query=args.get("query", args["ticker"]),
        )
        citations = "\n".join(ev.sources[:5]) if ev.sources else "(no citations)"
        return (
            f"signal={ev.signal:+.3f}  confidence={ev.confidence:.2f}\n"
            f"headline: {ev.headline}\n"
            f"details: {ev.details}\n"
            f"sources:\n{citations}"
        )

    def _risk(self, args):
        ev = self.orch.risk_agent.analyze(args["ticker"])
        return (
            f"signal={ev.signal:+.3f}  confidence={ev.confidence:.2f}\n"
            f"headline: {ev.headline}\n"
            f"details: {ev.details}"
        )

    def _rag(self, args):
        if self.rag is None:
            return "[rag unavailable: index not loaded]"
        from src.rag.retriever import format_hits
        top_k = int(args.get("top_k", 5))
        filter_stock = args.get("ticker")
        hits = self.rag.query(args["query"], top_k=top_k,
                              filter_stock=filter_stock)
        return format_hits(hits, include_content=False)

    def _prediction(self, args):
        from .prediction_reader import read_latest_prediction
        p = read_latest_prediction(args["ticker"])
        if not p.get("available"):
            return f"No prediction stored for {args['ticker']}."
        return (
            f"as_of={p['as_of']}  horizon={p['horizon']}\n"
            f"current_price={p['current_price']:.2f}  "
            f"predicted_price={p['predicted_price']:.2f}  "
            f"predicted_return={p['predicted_return']:+.2%}\n"
            f"fusion={p['fusion']}  model={p['model_type']}"
        )
