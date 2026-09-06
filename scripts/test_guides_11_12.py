"""
Comprehensive test: verify guide 11 (RAG) and guide 12 (multi-agent) work.

Runs a series of end-to-end checks and prints a pass/fail report.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


# ==================================================================
banner("GUIDE 11 — RAG SYSTEM")
# ==================================================================
from src.rag.retriever import NewsRetriever, format_hits

# Index files exist
check("FAISS index exists", (PROJECT_ROOT / "models/rag/faiss.index").exists())
check("Metadata parquet exists", (PROJECT_ROOT / "models/rag/metadata.parquet").exists())
check("Config pkl exists", (PROJECT_ROOT / "models/rag/config.pkl").exists())

r = NewsRetriever(index_dir=PROJECT_ROOT / "models/rag")

# English query
hits = r.query("BEXIMCO profit decline", top_k=3)
check("English query returns hits", len(hits) > 0, f"got {len(hits)}")
check("Top hit is BEXIMCO-related", "BEXIMCO" in hits[0].stock, f"got {hits[0].stock}")
check("Hits have similarity scores", all(0 <= h.similarity <= 1 for h in hits))

# Bangla query
bangla_hits = r.query("ব্যাংক ঋণ জালিয়াতি", top_k=3)
check("Bangla query returns hits", len(bangla_hits) > 0)
check("Bangla hits are Bangla articles", all(h.language == "bn" for h in bangla_hits))

# Stock filter
gp_hits = r.query("telecom", top_k=5, filter_stock="GP")
check("Stock filter (GP) works", all(h.stock == "GP" for h in gp_hits), f"got {len(gp_hits)} GP hits")

# Date filter
date_hits = r.query("dividend announcement", top_k=5, filter_date_from="2024-01-01")
check("Date filter (>=2024) works", all(h.date >= "2024-01-01" for h in date_hits), f"got {len(date_hits)} hits")

# Demo script ran
check("Demo output file exists", (PROJECT_ROOT / "results/rag/demo_queries.txt").exists())


# ==================================================================
banner("GUIDE 12 — MULTI-AGENT SYSTEM")
# ==================================================================
from src.agents import Orchestrator, AdvisorQuery, AgentEvidence

orch = Orchestrator(
    models_dir=PROJECT_ROOT / "models/baseline",
    data_dir=PROJECT_ROOT / "data/processed",
    sentiment_csv=PROJECT_ROOT / "results/sentiment/news_scored.csv",
    rag_index_dir=PROJECT_ROOT / "models/rag",
)

# Parse query
q = AdvisorQuery(user_query="Should I buy BATBC?")
q = orch.parse_query(q)
check("parse_query extracts BATBC", q.stock == "BATBC")
check("Capital stays default", q.capital_bdt == 100_000.0)

# Profile parsing
q2 = AdvisorQuery(user_query="Conservative view on GP with 5 lakh taka")
q2 = orch.parse_query(q2)
check("parse_query extracts GP", q2.stock == "GP")
check("parse_query extracts conservative", q2.user_profile == "conservative")
check("parse_query extracts 5 lakh → 500000", q2.capital_bdt == 500_000.0, f"got {q2.capital_bdt}")

# Advise — should not crash
result = orch.advise(AdvisorQuery(user_query="Tell me about SQURPHARMA"))
check("advise returns AdvisorResult", result is not None)
check("result has stock", result.stock == "SQURPHARMA")
check("result has recommendation", result.recommendation in ("BUY", "SELL", "HOLD"))
check("result has 5 reasoning bullets", len(result.reasoning) == 5, f"got {len(result.reasoning)}")
check("ensemble signal in range", -1.0 <= result.ensemble_signal <= 1.0)
check("confidence in range", 0.0 <= result.confidence <= 1.0)
check("consensus label is valid", result.consensus in ("unanimous", "majority", "split", "none"))
check("4 agents contributed", len(result.agents_used) == 4)
check("4 evidence packets stored", len(result.raw_evidence) == 4)

# Rank-all
df = orch.rank_all(profile="moderate", capital_bdt=100_000.0, top_k=10)
check("rank_all returns DataFrame", hasattr(df, "columns"))
check("rank_all has 10 rows", len(df) == 10, f"got {len(df)}")
check("rank_all sorted by signal", list(df["ensemble_signal"]) == sorted(df["ensemble_signal"], reverse=True))
check("rank_all has direction column", "direction" in df.columns)
check("top stocks are BUY/HOLD", all(d in ("BUY", "HOLD") for d in df["direction"]))

# Unknown stock handling
unknown = orch.advise(AdvisorQuery(user_query="What about some random thing?"))
check("Unknown stock → HOLD + 0 confidence", unknown.recommendation == "HOLD" and unknown.confidence == 0.0)

# Evidence packet structure
ev = result.raw_evidence[0]
check("Evidence has agent_name", isinstance(ev.agent_name, str))
check("Evidence has signal", isinstance(ev.signal, float))
check("Evidence has confidence", isinstance(ev.confidence, float))
check("Evidence has headline", isinstance(ev.headline, str))
check("Evidence has details dict", isinstance(ev.details, dict))

# Agent evidence direction property
sample_ev = AgentEvidence(agent_name="test", stock="X", signal=0.5, confidence=0.5, headline="")
check("Direction property BUY at +0.5", sample_ev.direction == "BUY")
sample_ev.signal = -0.5
check("Direction property SELL at -0.5", sample_ev.direction == "SELL")
sample_ev.signal = 0.0
check("Direction property HOLD at 0.0", sample_ev.direction == "HOLD")


# ==================================================================
banner(f"SUMMARY: {passed} passed, {failed} failed")
# ==================================================================
if failed:
    sys.exit(1)
