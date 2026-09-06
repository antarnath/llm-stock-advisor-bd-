"""
Orchestrator — coordinates the multi-agent system.

Workflow:
  1. Parse user query → identify target stock(s), user profile, capital
  2. Run Technical, News, Risk agents in parallel (collect evidence packets)
  3. Combine evidence with confidence-weighted average → ensemble signal
  4. Run Portfolio agent with ensemble signal → position sizing
  5. Synthesize final recommendation with reasoning chain
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Agent, AgentEvidence
from .technical_agent import TechnicalAgent
from .news_agent import NewsAgent
from .risk_agent import RiskAgent
from .portfolio_agent import PortfolioAgent


# Stock universe (the 30 DSE stocks this project covers)
STOCK_UNIVERSE = [
    "ACI", "BANKASIA", "BATBC", "BEXIMCO", "BEXPHARMA", "BRACBANK",
    "BSCCL", "CUSTOMERS", "DBBL", "DUTCHBANGL", "EBL", "GP",
    "HEIDELBCEM", "ISLAMI BANK", "JAMUNAOIL", "LAFARGECEM", "MARICO",
    "MUTUALTRUST", "NCCBANK", "POWERGRID", "PRIMEBANK", "RENATA",
    "ROBI", "SIBL", "SQURPHARMA", "SUMITPOWER", "TITASGAS",
    "UNILEVER", "WALTONHIL",
]


@dataclass
class AdvisorQuery:
    """User's request to the advisor."""
    user_query: str
    stock: Optional[str] = None       # explicit stock if user named it
    user_profile: str = "moderate"    # conservative / moderate / aggressive
    capital_bdt: float = 100_000.0    # user's investable capital
    top_k_news: int = 5


@dataclass
class AdvisorResult:
    """Synthesized final recommendation."""
    stock: str
    recommendation: str               # BUY / SELL / HOLD
    confidence: float                 # 0.0 - 1.0
    ensemble_signal: float            # -1.0 - +1.0
    position_pct: float               # fraction of capital to allocate
    position_bdt: float               # absolute position size
    reasoning: list[str]              # bullet points, one per agent
    agents_used: list[str]            # which agents contributed
    consensus: str                    # "unanimous" / "majority" / "split"
    raw_evidence: list[AgentEvidence] = field(default_factory=list)


class Orchestrator:
    """Coordinates the multi-agent financial advisory system."""

    def __init__(self,
                 models_dir: Path = Path("models/baseline"),
                 data_dir: Path = Path("data/processed"),
                 sentiment_csv: Path = Path("results/sentiment/news_scored.csv"),
                 rag_index_dir: Path = Path("models/rag"),
                 default_profile: str = "moderate"):
        self.technical_agent = TechnicalAgent(models_dir=models_dir, data_dir=data_dir)
        self.news_agent = NewsAgent(sentiment_csv=sentiment_csv, rag_index_dir=rag_index_dir)
        self.risk_agent = RiskAgent(data_dir=data_dir)
        self.portfolio_agent = PortfolioAgent(default_profile=default_profile)

    # ------------------------------------------------------------------
    # Query parsing
    # ------------------------------------------------------------------
    def parse_query(self, q: AdvisorQuery) -> AdvisorQuery:
        """Best-effort: extract stock code from user_query if not set."""
        if q.stock:
            return q
        text = q.user_query.upper()
        # Try each known stock as a substring (longest first to avoid false hits)
        for stock in sorted(STOCK_UNIVERSE, key=len, reverse=True):
            if re.search(rf"\b{stock}\b", text) or stock in text:
                q.stock = stock
                break
        # Extract profile
        low = q.user_query.lower()
        if "conservative" in low:
            q.user_profile = "conservative"
        elif "aggressive" in low:
            q.user_profile = "aggressive"
        elif "moderate" in low or "balanced" in low:
            q.user_profile = "moderate"
        # Extract capital
        m = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:lakh|lac|tk|bdt|taka)", low)
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
                if "lakh" in low or "lac" in low:
                    num *= 100_000
                q.capital_bdt = num
            except Exception:
                pass
        return q

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def advise(self, query: AdvisorQuery) -> AdvisorResult:
        """Run all agents and synthesize a final recommendation."""
        query = self.parse_query(query)
        if not query.stock:
            return AdvisorResult(
                stock="UNKNOWN",
                recommendation="HOLD",
                confidence=0.0,
                ensemble_signal=0.0,
                position_pct=0.0,
                position_bdt=0.0,
                reasoning=["Could not identify a stock in your query. "
                           "Please name one of the 30 supported DSE stocks."],
                agents_used=[],
                consensus="none",
            )

        stock = query.stock

        # Run three specialist agents (parallel possible; here sequential for simplicity)
        tech_ev = self.technical_agent.analyze(stock)
        news_ev = self.news_agent.analyze(stock, query=query.user_query)
        risk_ev = self.risk_agent.analyze(stock)

        # Confidence-weighted ensemble
        evidence_list = [tech_ev, news_ev, risk_ev]
        weights = np.array([tech_ev.confidence, news_ev.confidence, risk_ev.confidence])
        signals = np.array([tech_ev.signal, news_ev.signal, risk_ev.signal])
        if weights.sum() > 0:
            ensemble_signal = float((signals * weights).sum() / weights.sum())
            ensemble_conf = float(np.clip(np.average(
                [tech_ev.confidence, news_ev.confidence, risk_ev.confidence],
                weights=[tech_ev.confidence, news_ev.confidence, risk_ev.confidence],
            ), 0.0, 1.0))
        else:
            ensemble_signal = 0.0
            ensemble_conf = 0.0

        # Portfolio sizing
        port_ev = self.portfolio_agent.analyze(
            stock=stock,
            combined_signal=ensemble_signal,
            combined_confidence=ensemble_conf,
            capital_bdt=query.capital_bdt,
            user_profile=query.user_profile,
        )

        # Build consensus label
        directions = [tech_ev.direction, news_ev.direction, risk_ev.direction]
        if directions.count(directions[0]) == 3:
            consensus = "unanimous"
        elif len(set(directions)) == 2:
            consensus = "majority"
        else:
            consensus = "split"

        # Synthesize reasoning
        reasoning = [
            f"📊 Technical: {tech_ev.headline}",
            f"📰 News:      {news_ev.headline}",
            f"⚠️  Risk:      {risk_ev.headline}",
            f"💼 Portfolio: {port_ev.headline}",
            f"🤝 Consensus: {consensus}  (ensemble signal={ensemble_signal:+.2f}, "
            f"confidence={ensemble_conf:.0%})",
        ]

        # Final recommendation
        recommendation = port_ev.details.get("direction", "HOLD")

        return AdvisorResult(
            stock=stock,
            recommendation=recommendation,
            confidence=ensemble_conf,
            ensemble_signal=ensemble_signal,
            position_pct=port_ev.details.get("position_pct", 0.0),
            position_bdt=port_ev.details.get("position_bdt", 0.0),
            reasoning=reasoning,
            agents_used=[tech_ev.agent_name, news_ev.agent_name, risk_ev.agent_name, port_ev.agent_name],
            consensus=consensus,
            raw_evidence=evidence_list + [port_ev],
        )

    # ------------------------------------------------------------------
    # Batch: rank all 30 stocks by ensemble signal
    # ------------------------------------------------------------------
    def rank_all(self, profile: str = "moderate",
                 capital_bdt: float = 100_000.0,
                 top_k: int = 10):
        """Score every stock and return a ranked DataFrame."""
        rows = []
        for stock in STOCK_UNIVERSE:
            try:
                tech_ev = self.technical_agent.analyze(stock)
                news_ev = self.news_agent.analyze(stock, query=stock)
                risk_ev = self.risk_agent.analyze(stock)
                weights = np.array([tech_ev.confidence, news_ev.confidence, risk_ev.confidence])
                signals = np.array([tech_ev.signal, news_ev.signal, risk_ev.signal])
                if weights.sum() > 0:
                    ens_sig = float((signals * weights).sum() / weights.sum())
                    ens_conf = float(weights.mean())
                else:
                    ens_sig, ens_conf = 0.0, 0.0
                port_ev = self.portfolio_agent.analyze(
                    stock=stock, combined_signal=ens_sig,
                    combined_confidence=ens_conf,
                    capital_bdt=capital_bdt, user_profile=profile,
                )
                rows.append({
                    "stock": stock,
                    "ensemble_signal": ens_sig,
                    "confidence": ens_conf,
                    "tech_signal": tech_ev.signal,
                    "news_signal": news_ev.signal,
                    "risk_signal": risk_ev.signal,
                    "direction": port_ev.details.get("direction", "HOLD"),
                    "position_pct": port_ev.details.get("position_pct", 0.0),
                    "position_bdt": port_ev.details.get("position_bdt", 0.0),
                })
            except Exception as e:
                rows.append({"stock": stock, "error": str(e)})
        import pandas as pd
        df = pd.DataFrame(rows)
        if "ensemble_signal" in df.columns:
            df = df.sort_values("ensemble_signal", ascending=False).reset_index(drop=True)
        return df.head(top_k)
