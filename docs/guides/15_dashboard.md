# Dashboard — DSE Advisor

**Duration**: 1 Week
**Started**: Week 25
**Status**: 🚧 In progress — backend + frontend being rebuilt; this guide describes the target design and current state.

---

## 🎯 What This Dashboard Is

A web UI that puts the entire LLM-orchestrated financial advisor in front of a Bangladesh retail investor. The user types a question in plain English (or Bangla), sees a citation-rich answer, can drill into any of the 30 DSE stocks, build a portfolio with a slider for capital and risk profile, and configure which LLM provider powers the chat.

Everything the dashboard shows is already produced by `src/` — the dashboard is just a presentation layer that calls into:

| Backend module | What it gives the dashboard |
|---|---|
| `src.orchestrator.Orchestrator` | Chat answers with multi-agent evidence + RAG citations |
| `src.orchestrator.prediction_reader.read_predictions` | Latest multimodal 5-day price prediction |
| `src.orchestrator.impact_reader.read_recent_impact` | Recent news with bullish/bearish label |
| `src.orchestrator.freshness.freshness` | How stale the news corpus is |
| `src.portfolio.mean_variance.from_agent_signals` | Optimal portfolio from agent views |
| `src.portfolio.risk_parity.risk_parity_weights` | Equal-risk-contribution portfolio |
| `src.utils.config.TOP_30_DSE_STOCKS` + `load_stock_meta()` | Stock list + name + sector |

---

## 👤 Who the User Is

A Bangladeshi retail investor who:

- Knows ticker codes (GP, BATBC, BEXIMCO) and some company names
- Has ৳50,000–৳500,000 to invest
- Wants a second opinion before clicking Buy/Sell
- Reads news in Bangla or English

The dashboard assumes **zero finance PhD**. Every chart, number, and recommendation has a plain-language label next to it.

---

## 🖼️ Pages — What the User Sees

### Page 1 — `/` Landing

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  DSE Advisor                                          │
│  Bangladesh stock market in plain language.          │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  Should I buy GP next week?                    │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  Tip: try  GP  ·  BEXIMCO  ·  SQURPHARMA             │
│  Or configure the LLM provider.                       │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Type a question, press Enter → routed to `/chat?q=...`.

---

### Page 2 — `/chat?q=...` Chat Answer

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  ← back                                              │
│                                                       │
│  Q: Should I buy GP next week?                       │
│  Resolved: GP  ·  mode: template  ·  tools: agent,rag│
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ Analysis for GP:                                │  │
│  │                                                 │  │
│  │ Ticker: GP                                      │  │
│  │ Recommendation: HOLD  (confidence 68%,          │  │
│  │   consensus unanimous)                          │  │
│  │ Ensemble signal: +0.090                         │  │
│  │ Position size: 0.0%  (৳0)                       │  │
│  │                                                  │  │
│  │ Reasoning chain:                                │  │
│  │   📊 Technical: Model predicts GP up 0.09%      │  │
│  │   📰 News: positive (weighted_score=+0.12,       │  │
│  │        20 articles)                              │  │
│  │   ⚠️  Risk: vol=1.73%/day, Sharpe=-1.05           │  │
│  │   💼 Portfolio: HOLD GP, allocate 0.0%            │  │
│  │   🤝 Consensus: unanimous                        │  │
│  │                                                  │  │
│  │ Relevant news:                                   │  │
│  │ [1] 2026-08-25 | GP | sim=0.89                   │  │
│  │     BATBC declares dividend, sector up            │  │
│  │ ...                                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ ↻ Ask again                                      │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Read the recommendation, drill into one of the cited tickers (clickable), or rerun the same question.

---

### Page 3 — `/ticker/GP` Stock Detail

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  GP                                                   │
│  Grameenphone  ·  sector: Telecom                     │
│                                                       │
│  Current price: ৳194.71                              │
│  ─────────────────────────────────────────────────    │
│                                                       │
│  ┌── 5-day prediction ────────────────────────────┐  │
│  │ Predicted close: ৳194.15                        │  │
│  │ Predicted return: 0.00%                          │  │
│  │ Model: multimodal_late                          │  │
│  │ Fusion strategy: late                            │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌── Last 7 days of news ─────────────────────────┐  │
│  │ • 2026-08-22  BULLISH  magnitude 0.78           │  │
│  │   "GP reports strong Q2 earnings"               │  │
│  │ • 2026-08-19  BEARISH  magnitude 0.41           │  │
│  │   "BTRC tightens spectrum rules"                │  │
│  │ • ...                                            │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  [ 💬 Ask advisor about GP ]                          │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Cross-check the latest multimodal prediction with recent news, then click "Ask advisor" to get a full multi-agent breakdown.

---

### Page 4 — `/portfolio` Portfolio Builder

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  Build your portfolio                                 │
│                                                       │
│  Capital:  [ ৳100,000        ]  (slider 50K–500K)    │
│  Profile:  ( ) Conservative                          │
│            (•) Moderate                               │
│            ( ) Aggressive                              │
│                                                       │
│  Method:  (•) Max Sharpe  ( ) Min Variance            │
│           ( ) Risk Parity                            │
│                                                       │
│  [ Compute optimal portfolio ]                       │
│                                                       │
│  ─── Result ─────────────────────────────────────    │
│  Expected return:  12.4% / yr                        │
│  Volatility:       8.6% / yr                          │
│  Sharpe ratio:     1.31                                │
│                                                       │
│  ┌── Allocation ─────────────────────────────────┐  │
│  │ EBL         10.0%   ৳10,000                    │  │
│  │ POWERGRID   10.0%   ৳10,000                    │  │
│  │ SQURPHARMA  10.0%   ৳10,000                    │  │
│  │ UNILEVER    10.0%   ৳10,000                    │  │
│  │ BEXPHARMA   10.0%   ৳10,000                    │  │
│  │ GP           9.4%   ৳9,440                     │  │
│  │ RENATA       8.9%   ৳8,860                     │  │
│  │ DSEX         5.9%   ৳5,860                     │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Drag the capital slider, pick a risk profile + method, hit Compute. The table shows how to split the money. Per-stock cap is auto-applied from profile (5% / 10% / 20%).

---

### Page 5 — `/settings` LLM Provider

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  LLM Provider                                         │
│                                                       │
│  Provider:   [ openrouter          ▼ ]               │
│  Model:      [ meta-llama/llama-3.1-8b-instruct:free ] │
│  Base URL:   [ https://openrouter.ai/api/v1         ] │
│  API key:    [ •••••••••••••••••••••• ]              │
│  Has key:    ✓                                        │
│                                                       │
│  [ Save ]   [ Test connection ]                      │
│                                                       │
│  Available providers:                                │
│   • stub       — no key needed, deterministic         │
│   • openrouter — free + paid models, OPENROUTER_API_KEY│
│   • groq       — fast Llama, GROQ_API_KEY              │
│   • gemini     — Gemini 1.5 Flash, GOOGLE_API_KEY      │
│   • nim        — NVIDIA NIM endpoint, NIM_API_KEY      │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Pick a provider, paste a key, save. The next `/chat` call uses it. If no key is set, the system falls back to the template (deterministic, always works).

---

### Page 6 — `/news` News Freshness

**What the user sees:**

```
┌──────────────────────────────────────────────────────┐
│  News corpus                                          │
│                                                       │
│  Last refresh:  2026-09-04 18:32 UTC (3.2h ago)      │
│  Status:        ✓ ok                                  │
│  Articles added: 42                                    │
│  Stocks updated: GP, BATBC, BEXIMCO, RENATA, ...      │
│                                                       │
│  [ Trigger refresh now ]                              │
└──────────────────────────────────────────────────────┘
```

**How they use it:** Verify the corpus isn't stale before trusting news citations in a `/chat` answer. Trigger a refresh manually.

---

## 🔌 Backend API (FastAPI — to be recreated)

| Method | Path | Returns |
|---|---|---|
| `POST` | `/ask` | `Orchestrator.answer(question)` reply + metadata |
| `GET` | `/stocks/{ticker}` | price + name + sector |
| `GET` | `/stocks/{ticker}/prediction` | `read_predictions(ticker)` |
| `GET` | `/stocks/{ticker}/news` | `read_recent_impact(ticker, days=7)` |
| `POST` | `/portfolio/optimize` | `from_agent_signals(views, capital, profile)` |
| `GET / POST` | `/settings` | LLM provider config |
| `GET` | `/freshness` | `freshness()` |
| `GET` | `/healthz` | `{"status":"ok"}` |

All endpoints return JSON. The frontend calls them via Next.js server components (no CORS, no exposed API keys).

---

## 🧱 Tech Stack

```
Frontend  : Next.js 14 (App Router), TypeScript, Tailwind CSS
Backend   : FastAPI, Pydantic, uvicorn
ML        : src/orchestrator, src/portfolio, src/rag, src/llm
Data      : data/processed/*_processed_v2.csv, results/multimodal/*.csv
```

---

## 🛠️ How It Runs

```bash
# Backend (terminal 1)
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Frontend (terminal 2)
cd frontend && npm install && BACKEND_URL=http://localhost:8000 npm run dev
# → http://localhost:3000
```

---

## 🚧 Current Build Status

| Component | Status |
|---|---|
| Backend folder | ❌ Deleted; will recreate (lean version, ~400 lines) |
| Frontend folder | ❌ Deleted; will recreate (5 pages, ~700 lines) |
| Backend ↔ orchestrator wiring | verified working (live-tested before deletion) |
| Frontend ↔ backend wiring | verified working (live-tested before deletion) |
| Dashboard spec (this doc) | ✅ written |

---

## 📋 Page Implementation Order

1. `/healthz` + `/` landing → confirm stack up
2. `/chat` → most important page, full evidence pack visible
3. `/ticker/[code]` → second most important, prediction + news
4. `/portfolio` → builder with slider + profile + method
5. `/settings` → provider dropdown + key
6. `/news` → freshness display + manual trigger

---

**Last Updated**: 2026-09-06
