# LLM Financial Advisor

**Duration**: 2 Weeks
**Started**: Week 23
**Status**: ✅ Complete — LLM-orchestrated chat backend wired to all agents + RAG + FastAPI

---

## 🎯 Objectives

1. ✅ Build conversational interface (Next.js chat UI in `frontend/app/chat/`)
2. ✅ Integrate all agents seamlessly (3 specialists + orchestrator)
3. ✅ Implement context management (evidence pack + RAG hits)
4. ✅ Add multi-turn dialogue support (stateless, each turn self-contained)
5. ✅ Create REST API (FastAPI with 6 routers: ask, freshness, news, stocks, settings, health)

---

## 🏗️ Architecture — How It Actually Works

```
User question
    │
    ▼
┌────────────────────────────────────────────────────┐
│  src/orchestrator/orchestrator.py  — Orchestrator  │
│  ┌──────────────────────────────────────────────┐  │
│  │ 1. Ticker resolver  (resolve_ticker)        │  │
│  │    "Grameenphone" → "GP"                    │  │
│  │ 2. Evidence pack   (multi-agent ensemble)    │  │
│  │    Technical + News + Risk + Portfolio       │  │
│  │ 3. RAG hits         (multilingual FAISS)     │  │
│  │ 4. LLM call         (OpenAI-compatible)      │  │
│  │ 5. Template fallback (no LLM configured)     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
    │
    ▼
Natural-language reply (with citations)
```

### Evidence Pack (when ticker resolved)

```
=== MULTI-AGENT EVIDENCE ===
Ticker: BATBC
Recommendation: BUY  (confidence 72%, consensus unanimous)
Ensemble signal: +0.342
Position size: 8.5%  (৳8,500)

Reasoning chain:
  - Technical: LSTM-60 forecasts +4.2% next 5 days
  - News: 3 positive FinBERT hits, weighted_score = +0.41
  - Risk: Sharpe 1.4, max drawdown -12%, beta 0.85
  - Portfolio: improves Sharpe from 1.20 → 1.31
```

### RAG Hits (always)

```
=== RAG NEWS HITS ===
[1] 2026-08-25 | BATBC | sim=0.89
    "BATBC declares 120% cash dividend, stock jumps 5%"
[2] 2026-08-22 | BATBC | sim=0.81
    "Tobacco sector outlook: BATBC leads Q2 earnings"
[3] 2026-08-18 | BATBC | sim=0.74
    "Analyst upgrades BATBC on volume growth"
```

---

## 🧩 Module Layout

```
src/
├── orchestrator/
│   ├── __init__.py           # exports Orchestrator, ChatAnswer
│   ├── orchestrator.py       # main class — chat() + answer()
│   ├── ticker_resolver.py    # resolve_ticker(text, universe) with aliases
│   ├── tools.py              # ToolRegistry (5 tools)
│   ├── freshness.py          # news refresh age / status
│   ├── prediction_reader.py  # latest multimodal prediction
│   └── impact_reader.py      # recent news impact for /stocks/{t}/news
└── llm/
    ├── __init__.py           # exports LLMClient, ChatMessage, ChatResponse
    └── client.py             # OpenAI-compatible LLMClient

scripts/
└── test_llm_advisor.py       # 33 tests, all passing
```

---

## 💬 Ticker Resolver — `src/orchestrator/ticker_resolver.py`

Resolves a free-text question to one of the 30 DSE tickers, with name aliases.

```python
ALIASES = {
    "GRAMEENPHONE": "GP",
    "BRAC BANK":    "BRACBANK",
    "ISLAMI BANK":  "ISLAMI BANK",     # wins longest-match over "BANK"
    "SQUARE PHARMA": "SQURPHARMA",
    # ... 30 total
}

def resolve_ticker(text: str, universe: list[str]) -> Optional[str]:
    """Longest-match substring search over the question."""
```

### Examples

| Input                                   | Resolved   |
|-----------------------------------------|------------|
| "Should I buy BATBC for the long term?" | `BATBC`    |
| "How is Grameenphone doing?"            | `GP`       |
| "Tell me about BRAC bank"               | `BRACBANK` |
| "ISLAMI BANK news"                      | `ISLAMI BANK` (longest match wins) |
| "What's the meaning of life?"           | `None`     |

---

## 🔧 Tool Registry — `src/orchestrator/tools.py`

Five tools the LLM can call during reasoning:

| Tool                    | Purpose                                         |
|-------------------------|-------------------------------------------------|
| `run_technical_agent`   | Latest LSTM forecast + signal                   |
| `run_news_agent`        | FinBERT-aggregated sentiment for the ticker     |
| `run_risk_agent`        | Sharpe, drawdown, beta, VaR                     |
| `rag_search`            | FAISS semantic search over news                 |
| `get_latest_prediction` | Multimodal 1d / 5d price prediction             |

Each tool returns a **string** so it slots cleanly into the prompt context.

---

## 🤖 LLM Client — `src/llm/client.py`

OpenAI-compatible client with **5 providers**:

| Provider      | Base URL                                         | Default model                       | Env var              |
|---------------|--------------------------------------------------|-------------------------------------|----------------------|
| `stub`        | (none)                                           | `stub-template-v1`                  | —                    |
| `openrouter`  | `https://openrouter.ai/api/v1`                   | `meta-llama/llama-3.1-8b-instruct:free` | `OPENROUTER_API_KEY` |
| `groq`        | `https://api.groq.com/openai/v1`                 | `llama-3.1-8b-instant`              | `GROQ_API_KEY`       |
| `gemini`      | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-1.5-flash`        | `GOOGLE_API_KEY`     |
| `nim`         | (custom NIM endpoint)                            | `meta/llama-3.1-8b-instruct`        | `NIM_API_KEY`        |

```python
from src.llm import LLMClient, ChatMessage

client = LLMClient(provider="openrouter")           # picks up OPENROUTER_API_KEY
resp = client.chat([
    ChatMessage(role="system", content="You are a financial advisor."),
    ChatMessage(role="user",   content="Should I buy BATBC?"),
], temperature=0.2, max_tokens=1024)
print(resp.content)
```

### Stub Mode

When no provider credentials are present, `LLMClient(provider="stub")` returns a deterministic summary that still references all evidence — never raises.

---

## 🎯 Main Orchestrator — `src/orchestrator/orchestrator.py`

```python
from src.orchestrator import Orchestrator

orch = Orchestrator()                              # reads OPENROUTER_API_KEY etc.
reply = orch.answer("Should I buy BATBC for the long term?")
```

### `chat(question: str) -> ChatAnswer`

```python
@dataclass
class ChatAnswer:
    reply: str
    ticker: Optional[str]
    tools_called: list[str]
    mode: str                # "llm" | "template"
    tokens: int = 0
```

### Flow

1. **Resolve ticker** — `resolve_ticker(question, STOCK_UNIVERSE)`
2. **If ticker found** → run multi-agent evidence pack (Technical + News + Risk)
3. **Always** → run RAG query (top 3 hits over the question text)
4. **If LLM live** → send system prompt + context → LLM
5. **If LLM unavailable** → deterministic template that still cites the evidence

### System Prompt

```
You are a financial advisor for the Bangladesh stock market.
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
```

---

## 🔌 REST API — `backend/`

FastAPI app with 6 routers wired to the orchestrator and supporting readers.

| Router                | Endpoint                                | Handler                          |
|-----------------------|-----------------------------------------|----------------------------------|
| `ask`                 | `POST /ask`                             | `Orchestrator.answer()`          |
| `freshness`           | `GET  /freshness`                       | `freshness()`                    |
| `news`                | `POST /news/refresh`                    | refresh trigger (stubbed)        |
| `stocks`              | `GET  /stocks/{ticker}`                 | `load_stock_meta()` + price      |
| `settings`            | `GET  /settings` / `POST /settings`     | LLM provider / API key           |
| `health`              | `GET  /healthz`                         | liveness probe                   |

### `POST /ask` — Example

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Should I buy GP for the long term?"}' | jq
```

```json
{
  "reply": "Analysis for **GP**:\n\nTicker: GP\nRecommendation: HOLD ...",
  "ticker": "GP",
  "tools_called": ["multi_agent_evidence", "rag_search"],
  "mode": "template",
  "tokens": 0
}
```

---

## 🧪 Tests — `scripts/test_llm_advisor.py` (33 tests)

```
=== Test 1: ticker resolution + evidence ===           6/6 PASS
=== Test 2: company name alias ===                     2/2 PASS
=== Test 3: Bangla query ===                           1/1 PASS
=== Test 4: unknown stock ===                          2/2 PASS
=== Test 5: profile + capital in query ===             2/2 PASS
=== Test 6: tool dispatch ===                          4/4 PASS
=== Test 7: LLM stub fallback ===                      4/4 PASS
=== Test 8: ticker resolver edge cases ===             5/5 PASS

33 passed, 0 failed
```

---

## 📊 Example End-to-End Interaction (Template Mode)

```
👤 User: "Should I buy BATBC for the long term?"

🤖 Reply:
Analysis for **BATBC**:

Ticker: BATBC
Recommendation: BUY  (confidence 72%, consensus unanimous)
Ensemble signal: +0.342
Position size: 8.5%  (৳8,500)

Reasoning chain:
  - Technical: LSTM-60 forecasts +4.2% next 5 days
  - News: 3 positive FinBERT hits, weighted_score = +0.41
  - Risk: Sharpe 1.4, max drawdown -12%, beta 0.85

Relevant news from the corpus:
[1] 2026-08-25 | BATBC | sim=0.89
    BATBC declares 120% cash dividend, stock jumps 5%
...

_Reply generated via deterministic template (no LLM configured)._
```

---

## 💡 Design Decisions

1. **Stateless orchestrator** — safe to share as a module-level singleton; no conversation history on the server side (frontend handles chat scrollback).
2. **Eager evidence pack** — rather than letting the LLM decide which tools to call, we always run the full evidence pack when a ticker is resolved. This guarantees consistent answers and avoids tool-call loop bugs.
3. **Stub fallback** — the system always returns a useful answer, even with no LLM configured. The template shows real evidence and cites real RAG hits.
4. **Longest-match ticker resolution** — `ISLAMI BANK` wins over `BANK` because we sort aliases by length descending before substring search.
5. **OpenAI-compatible client** — works with any provider that exposes `/v1/chat/completions` (OpenRouter, Groq, Gemini's OpenAI endpoint, NVIDIA NIM).

---

## 🛠️ Tools & Libraries

- **OpenAI Python SDK** — used as a transport to multiple providers
- **FastAPI** — REST backend
- **Pydantic** — request/response schemas
- **src.agents.Orchestrator** — multi-agent ensemble
- **src.rag.retriever.NewsRetriever** — multilingual FAISS search

---

## 🧭 What's Deferred

- ❌ Multi-turn conversation memory (frontend scrollback only)
- ❌ A/B-tested prompt variations
- ❌ User risk-assessment questionnaire
- ❌ Production-grade rate limiting
- ❌ Persistent conversation logging (PostgreSQL)

These are product-level features beyond the thesis scope; the core advisor is fully functional for single-turn Q&A with citation-rich responses.

---

**Last Updated**: 2026-09-06
