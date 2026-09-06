"""
LLM-Orchestrated Financial Advisor.

Top-level orchestrator that wraps the multi-agent system with an LLM layer
that translates evidence packets into natural-language explanations.

Components:
  - Orchestrator: main entry point with .answer(question) method
  - Tool dispatch: lets the LLM invoke specialist agents + RAG on demand
  - LLM client: pluggable OpenAI-compatible backend (OpenRouter, Groq,
                Gemini, NIM, or stub)
  - Ticker resolver: maps natural-language stock names to DSE codes
  - Impact reader: pulls recent news impact for a ticker

The chat loop:
  1. User asks a question
  2. Orchestrator resolves the ticker (if mentioned)
  3. LLM is given the user question + tool descriptions
  4. LLM may call tools (technical, news, risk, rag, portfolio)
  5. Tool results are folded back into the LLM context
  6. LLM produces final natural-language answer with citations

If no API key is configured, the orchestrator falls back to a
deterministic template-based answer using the rule-based agents
(no LLM call).
"""

from .orchestrator import Orchestrator, ChatAnswer

__all__ = ["Orchestrator", "ChatAnswer"]
