# Multi-Agent Financial Advisory System

## Overview

A four-specialist-agent architecture that converts heterogeneous market evidence
(price models, news sentiment, risk metrics, position sizing rules) into a
single actionable BUY / SELL / HOLD recommendation per stock.

```
┌──────────────────────────── Orchestrator ────────────────────────────┐
│                                                                       │
│  user_query  ──►  parse_query()  ──►  [Tech, News, Risk]  ──►  ens.   │
│                                                                       │
│  ens. + profile + capital  ──►  PortfolioAgent  ──►  final rec.      │
└───────────────────────────────────────────────────────────────────────┘
```

## Agents

| Agent | Role | Input | Output (`signal`, `confidence`) |
|---|---|---|---|
| `TechnicalAgent` | Price-only forecast | `{STOCK}_best_v2.pkl` + last 60 rows | `tanh(predicted_return / recent_vol)` |
| `NewsAgent` | News sentiment | `news_scored.csv` + RAG citations | weighted sentiment over last 20 articles |
| `RiskAgent` | Risk profile | last 252 daily returns | Sharpe + vol + drawdown composite |
| `PortfolioAgent` | Position sizing | ensemble + user profile | `position_pct`, `position_bdt`, direction |

## Evidence packet

Every agent emits an `AgentEvidence` dataclass:

```python
@dataclass
class AgentEvidence:
    agent_name: str
    stock: str
    signal: float          # -1.0 (sell) ... +1.0 (buy)
    confidence: float      # 0.0 ... 1.0
    headline: str          # 1-sentence summary
    details: dict          # agent-specific key-value evidence
    top_features: list     # top-3 contributing features
    sources: list          # file paths, RAG citations
```

## Orchestrator

### `parse_query()`
Best-effort regex extraction of:
- **Stock code** — substring match against 30-stock DSE universe (longest first).
- **Risk profile** — `conservative` / `moderate` / `aggressive`.
- **Capital (BDT)** — `1 lakh` → 100,000 BDT; raw integers in taka.

### `advise(query)`
1. Run `TechnicalAgent.analyze(stock)` → evidence.
2. Run `NewsAgent.analyze(stock, query=user_query)` → evidence (RAG optional).
3. Run `RiskAgent.analyze(stock)` → evidence.
4. Compute confidence-weighted ensemble signal.
5. Run `PortfolioAgent.analyze(...)` → final direction + sizing.
6. Label consensus (unanimous / majority / split).
7. Build reasoning chain (one bullet per agent + consensus summary).

### `rank_all(profile, capital, top_k)`
Scores every stock in the universe and returns the top-k by ensemble signal.

## Position sizing rules

| Profile | Max per stock | Cash reserve |
|---|---|---|
| Conservative | 5% | 60% |
| Moderate | 10% | 40% |
| Aggressive | 20% | 20% |

`position_pct = |ensemble_signal| × confidence × max_per_stock` (capped).

## Rule-based, not LLM-orchestrated

The orchestrator is deterministic and reproducible — no external LLM call.
Rationale:

- Latency — runs in seconds on CPU.
- Cost — zero API spend.
- Reproducibility — same query yields identical evidence chain.
- Auditability — every signal has a numeric source.

The LLM layer (planned for the advisor chat interface) wraps this orchestrator
and translates its output into natural-language guidance, rather than driving
the decision itself.

## Demo

```bash
.venv/bin/python scripts/run_advisor_demo.py
```

Runs four sample queries (long-term / conservative / aggressive / generic) and
prints the full reasoning chain, plus a top-10 rank-all table.
