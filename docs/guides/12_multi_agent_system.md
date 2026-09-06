# Multi-Agent Financial Advisory System

**Duration**: 2 Weeks
**Started**: Week 20
**Status**: ✅ Complete
**Goal**: Coordinate specialist agents into a single recommendation

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

## How It Works — Step by Step

### 1. Parse the user query

`Orchestrator.parse_query()` does best-effort extraction:

| Field | How it's extracted | Example |
|---|---|---|
| Stock code | Longest-match substring against the 30-stock universe | "Should I buy BATBC?" → `BATBC` |
| Risk profile | Keyword lookup (`conservative` / `moderate` / `aggressive` / `balanced`) | "I'm conservative" → `conservative` |
| Capital (BDT) | Regex on `N lakh / lac / taka / BDT` | "1 lakh taka" → 100,000 |

If no stock can be identified, the orchestrator returns a HOLD with 0 confidence and a "please name a stock" message.

### 2. Run each specialist agent

Each agent is a thin class with a single `analyze(stock, **kwargs) -> AgentEvidence` method.

**TechnicalAgent** (`src/agents/technical_agent.py`)
- Loads `models/baseline/{STOCK}_best_v2.pkl` (LR, RF, or XGBoost bundle).
- Takes the last row of `{STOCK}_processed_v2.csv` and runs `.predict()`.
- Computes 20-day volatility from recent returns.
- `signal = tanh(predicted_return / volatility)` — bounded to [-1, +1].
- `confidence = clip(0.6 - 8 × volatility, 0.2, 0.9)` — less volatile → more confident.

**NewsAgent** (`src/agents/news_agent.py`)
- Reads `results/sentiment/news_scored.csv` filtered to the stock.
- Picks the most recent 20 articles.
- Computes confidence-weighted mean sentiment.
- Optionally retrieves RAG citations via `NewsRetriever` for the user's query text.
- `signal = weighted_sentiment` (already in [-1, +1]).
- `confidence = clip(0.3 + 0.04 × n_articles + 0.3 × |score|, 0.1, 0.9)`.

**RiskAgent** (`src/agents/risk_agent.py`)
- Reads the last 252 daily returns from `{STOCK}_processed_v2.csv`.
- Computes volatility, downside volatility, max drawdown, annualized Sharpe.
- Combines three sub-signals:
  - `sharpe_signal` ∝ sharpe (positive Sharpe → positive)
  - `vol_signal` ∝ -volatility (high vol → negative)
  - `drawdown_signal` ∝ |max_drawdown| (deep DD → negative)
- Final: `signal = 0.5 × sharpe + 0.2 × vol + 0.3 × drawdown` (clipped).

**PortfolioAgent** (`src/agents/portfolio_agent.py`)
- Takes the **ensemble** signal + confidence (it has no own signal).
- Looks up the user's profile in `PROFILES`:
  | Profile | Max per stock | Cash reserve |
  |---|---|---|
  | Conservative | 5% | 60% |
  | Moderate | 10% | 40% |
  | Aggressive | 20% | 20% |
- `position_pct = |ensemble_signal| × confidence × max_per_stock` (capped).
- `position_bdt = (capital × (1 - cash_reserve)) × position_pct`.
- Direction label: BUY if signal > +0.2, SELL if < -0.2, else HOLD.

### 3. Ensemble the three signals

```python
weights  = [tech.confidence, news.confidence, risk.confidence]
signals  = [tech.signal,     news.signal,     risk.signal]
ensemble = (signals * weights).sum() / weights.sum()
```

Confidence-weighted so a high-confidence specialist gets more say.

### 4. Synthesize the recommendation

The orchestrator returns an `AdvisorResult` with:

- `stock`, `recommendation` (BUY / SELL / HOLD)
- `ensemble_signal`, `confidence`
- `position_pct`, `position_bdt`
- `reasoning` (5-bullet chain: Tech, News, Risk, Portfolio, Consensus)
- `consensus` label:
  - `unanimous` — all 3 agents agree
  - `majority` — 2 of 3 agree
  - `split` — all 3 disagree
- `raw_evidence` — full evidence packets for auditability

## Code Layout

```
src/agents/
├── __init__.py             # exports Orchestrator, AdvisorQuery, AdvisorResult
├── base.py                 # Agent ABC + AgentEvidence dataclass
├── technical_agent.py
├── news_agent.py
├── risk_agent.py
├── portfolio_agent.py
└── orchestrator.py         # Orchestrator + AdvisorQuery + AdvisorResult

scripts/
└── run_advisor_demo.py     # 4 sample queries + top-10 rank-all
```

## Orchestrator API

### `parse_query(q: AdvisorQuery) -> AdvisorQuery`

### `advise(q: AdvisorQuery) -> AdvisorResult`

Full pipeline: parse → 3 agents → ensemble → portfolio agent → synth.

### `rank_all(profile, capital_bdt, top_k) -> pd.DataFrame`

Scores every stock in the universe and returns the top-k by ensemble signal.

## Rule-based, not LLM-orchestrated

The orchestrator is **deterministic and reproducible** — no external LLM call.
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

### Verified output (sample)

```
>>> USER: Aggressive trader here, looking at BEXIMCO with 1 lakh taka
────────────────────────────────────────────────────────────────────────
Stock: BEXIMCO    Recommendation: HOLD    Confidence: 68%
Ensemble signal: -0.03    Position: 0.0%  (৳0)
Consensus: split
Reasoning chain:
  • 📊 Technical: Model predicts BEXIMCO up 0.17% next day
  • 📰 News:      weighted_score=-0.27, 20 recent articles, 5 pos / 8 neg
  • ⚠️  Risk:      vol=2.85%/day, Sharpe=0.52, max DD=-33.6%
  • 💼 Portfolio: HOLD BEXIMCO under aggressive profile
  • 🤝 Consensus: split  (ensemble signal=-0.03, confidence=68%)
────────────────────────────────────────────────────────────────────────

=== RANK-ALL top 10 ===
  stock         ensemble_signal  confidence  direction
  UNILEVER      +0.31            0.64        BUY
  BEXPHARMA     +0.27            0.63        BUY
  EBL           +0.20            0.62        HOLD
  SQURPHARMA    +0.19            0.61        HOLD
  POWERGRID     +0.13            0.64        HOLD
  GP            +0.09            0.62        HOLD
  ...
```

---

**Last Updated**: 2026-09-06

