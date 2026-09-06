"""
Base classes for the multi-agent system.

Each agent produces an AgentEvidence packet. The Orchestrator collects
evidence from all agents and synthesizes a final recommendation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AgentEvidence:
    """Structured evidence packet produced by an agent."""
    agent_name: str
    stock: str
    # -1.0 (strong sell) to +1.0 (strong buy)
    signal: float
    # 0.0 (no confidence) to 1.0 (full confidence)
    confidence: float
    # Human-readable 1-sentence summary
    headline: str
    # Detailed evidence (key-value pairs the orchestrator can read)
    details: dict = field(default_factory=dict)
    # Top-3 contributing features (for SHAP-grounded explanations)
    top_features: list[tuple[str, float]] = field(default_factory=list)
    # Citations / sources used
    sources: list[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        if self.signal > 0.2:
            return "BUY"
        elif self.signal < -0.2:
            return "SELL"
        return "HOLD"


class Agent(ABC):
    """Base class for all specialist agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def analyze(self, stock: str, **kwargs) -> AgentEvidence:
        """Produce an evidence packet for the given stock."""
        ...
