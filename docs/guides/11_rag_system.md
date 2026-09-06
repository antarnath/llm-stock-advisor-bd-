# RAG System

**Duration**: 2 Weeks
**Started**: Week 18
**Status**: ✅ Complete
**Goal**: Build a retrieval-augmented generation index over the curated news corpus so the multi-agent system can cite real articles when answering questions.

---

## 🎯 What This Guide Built

A **multilingual semantic-search index** over the 1,560 curated DSE news
articles (English + Bangla) — used by the `NewsAgent` and the chat advisor to
ground natural-language answers in real article evidence.

The RAG layer sits *between* the news corpus and the LLM that will eventually
translate agent evidence into natural-language answers.

---

## 🔧 Technical Stack (what was actually built)

| Component | Choice | Why |
|---|---|---|
| Embedder | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, supports 50+ languages including Bangla, fast on CPU (~120MB) |
| Vector store | FAISS `IndexFlatIP` | Cosine similarity over L2-normalized vectors; no GPU needed |
| Metadata store | Parquet sidecar | news_id, date, stock, language, sentiment_score |
| Filters | post-hoc on top-`k × 20` candidate pool | acceptable for 1.5k docs |

No LangChain, no ChromaDB, no OpenAI embeddings — kept the stack minimal
to run on CPU within seconds.

---

## 📂 Code Layout

```
src/rag/
├── __init__.py        # exports NewsRetriever, retrieve, format_hits
├── embedder.py        # NewsEmbedder — wraps sentence-transformers
├── indexer.py         # NewsIndex — FAISS + metadata, build_index() helper
└── retriever.py       # NewsRetriever — friendly query interface

scripts/
├── build_news_index.py    # one-shot index builder
├── query_news.py          # CLI for ad-hoc queries
└── query_news_demo.py     # 8 curated demo queries (en + bn)

models/rag/
├── faiss.index        # ~2.3 MB FAISS binary
├── metadata.parquet   # article metadata
└── config.pkl         # {dim, n_docs}

results/rag/
└── demo_queries.txt   # 8 demo queries with retrieved hits
```

---

## 🚀 How It Works — Step by Step

### 1. Build the index

```bash
.venv/bin/python scripts/build_news_index.py
```

The script:

1. Reads `data/raw/news/news_curated.csv` (1,560 articles, columns:
   `news_id, date, stock, name, sector, language, headline, content`).
2. Joins with `results/sentiment/news_scored.csv` to attach sentiment
   score / label / confidence per article.
3. Concatenates `headline + ". " + content` to build the embedding text.
4. Loads `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers).
5. Encodes all 1,560 articles → (1560 × 384) float32 matrix, L2-normalized.
6. Builds FAISS `IndexFlatIP` (inner-product = cosine on normalized vectors).
7. Saves `faiss.index` + `metadata.parquet` + `config.pkl` to `models/rag/`.

### 2. Query the index

```python
from src.rag import NewsRetriever, format_hits

retriever = NewsRetriever(index_dir="models/rag/")
hits = retriever.query(
    "BEXIMCO profit decline",
    top_k=5,
    filter_stock="BEXIMCO",       # optional
    filter_language="en",         # optional
)
print(format_hits(hits, include_content=True))
```

Returns `list[NewsHit]` with: news_id, date, stock, name, sector, language,
headline, content, sentiment_score, sentiment_label, similarity, rank.

### 3. Use from the multi-agent system

The `NewsAgent` (in `src/agents/news_agent.py`) calls
`NewsRetriever.query(...)` and embeds the returned headlines as citations in
the agent evidence packet — so when the advisor says "BUY BEXPHARMA",
the user can see *which* news articles drove that signal.

---

## 🧪 Verification — Demo Queries

`scripts/query_news_demo.py` runs 8 curated queries covering both languages and
filter combinations. Output saved to `results/rag/demo_queries.txt`.

Example queries it tests:

| # | Query | Language | Filter |
|---|---|---|---|
| 1 | "BEXIMCO profit decline" | en | stock=BEXIMCO |
| 2 | "GP dividend announcement 2024" | en | stock=GP |
| 3 | "ব্যাংক ঋণ জালিয়াতি" (bank loan fraud) | bn | — |
| 4 | "বাংলাদেশ ব্যাংক সুদহার" (Bangladesh Bank interest rate) | bn | — |
| 5 | "BRAC BANK quarterly earnings" | en | stock=BRACBANK |
| 6 | "সিমেন্ট রপ্তানি" (cement export) | bn | sector=cement |
| 7 | "tobacco company new product launch" | en | sector=tobacco |
| 8 | "RENATA pharma expansion" | en | stock=RENATA |

Each query shows top-5 hits with similarity score + sentiment label.

---

## 🛠️ Tools & Libraries

- `sentence-transformers 3.2.1` — multilingual embedder (downgraded for
  PyTorch 2.2 compatibility)
- `faiss-cpu` — vector index
- `pandas` + `pyarrow` — metadata sidecar
- `numpy 1.x` — array ops (NumPy 2.x breaks old torch)

---

## ✅ Success Criteria

- [x] Index builder runs end-to-end on 1,560 articles
- [x] FAISS index persisted to `models/rag/faiss.index`
- [x] Multilingual — Bangla queries return Bangla articles
- [x] Metadata filters (stock / language / date range) work
- [x] NewsAgent integrates retriever into evidence packets
- [x] 8 demo queries reproducible via `query_news_demo.py`

---

## 🔬 Design Trade-offs

| Decision | Reason |
|---|---|
| FAISS over ChromaDB | Smaller, no server, perfect for static 1.5k corpus |
| `IndexFlatIP` (brute force) | Exact search; for 1.5k × 384, query is <1 ms |
| Post-hoc filtering | Avoids per-filter IVF training; fine at this scale |
| No LangChain | Added dependency, no benefit for our single-shot queries |
| Parquet sidecar | Faster + typed vs pickle; FAISS handles vectors only |
| Multilingual embedder (not OpenAI) | Avoids API cost; supports Bangla natively |

---

## 🚧 Limitations / Future Work

- No LLM re-ranking — relying on cosine similarity alone
- No query expansion or rewriting
- No incremental updates — full re-build required when new articles arrive
- No conversational memory (each query is independent)

---

**Last Updated**: 2026-09-06
