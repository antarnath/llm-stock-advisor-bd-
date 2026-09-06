"""
Multi-Agent Financial Advisory System.

A self-contained, rule-based multi-agent system where 4 specialist agents
(Technical, News, Risk, Portfolio) produce evidence packets, and an
Orchestrator synthesizes them into a final recommendation.

Design goals:
  - Deterministic (no external LLM) — reproducible for thesis
  - Each agent has a clear specialty
  - Orchestrator performs weighted vote + disagreement arbitration
  - All outputs include SHAP-grounded explanations + RAG-retrieved news
    context

Public API:
    from src.agents import Orchestrator, AdvisorQuery

    orch = Orchestrator()
    result = orch.advise(AdvisorQuery(
        user_query="Should I buy BATBC?",
        user_profile="moderate",
        capital_bdt=500_000,
    ))
    print(result.recommendation, result.confidence, result.reasoning)
"""

from .base import Agent, AgentEvidence
from .technical_agent import TechnicalAgent
from .news_agent import NewsAgent
from .risk_agent import RiskAgent
from .portfolio_agent import PortfolioAgent
from .orchestrator import Orchestrator, AdvisorQuery, AdvisorResult

__all__ = [
    "Agent", "AgentEvidence",
    "TechnicalAgent", "NewsAgent", "RiskAgent", "PortfolioAgent",
    "Orchestrator", "AdvisorQuery", "AdvisorResult",
]
