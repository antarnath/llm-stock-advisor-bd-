"""
Demo: end-to-end run of the multi-agent financial advisor.

Runs a few sample queries through the Orchestrator and prints the
synthesized recommendation + reasoning chain for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import Orchestrator, AdvisorQuery


def fmt_result(r) -> str:
    lines = [
        "─" * 72,
        f"Stock: {r.stock}    Recommendation: {r.recommendation}    "
        f"Confidence: {r.confidence:.0%}",
        f"Ensemble signal: {r.ensemble_signal:+.2f}    "
        f"Position: {r.position_pct*100:.1f}%  (৳{r.position_bdt:,.0f})",
        f"Consensus: {r.consensus}",
        "Reasoning chain:",
    ]
    for bullet in r.reasoning:
        lines.append(f"  • {bullet}")
    lines.append("─" * 72)
    return "\n".join(lines)


def main():
    project_root = Path(__file__).resolve().parents[1]
    orch = Orchestrator(
        models_dir=project_root / "models" / "baseline",
        data_dir=project_root / "data" / "processed",
        sentiment_csv=project_root / "results" / "sentiment" / "news_scored.csv",
        rag_index_dir=project_root / "models" / "rag",
    )

    queries = [
        AdvisorQuery(
            user_query="Should I buy BATBC for the long term?",
            user_profile="moderate",
            capital_bdt=200_000.0,
        ),
        AdvisorQuery(
            user_query="I'm conservative — what's your view on GP?",
            user_profile="conservative",
            capital_bdt=500_000.0,
        ),
        AdvisorQuery(
            user_query="Aggressive trader here, looking at BEXIMCO with 1 lakh taka",
            user_profile="aggressive",
            capital_bdt=100_000.0,
        ),
        AdvisorQuery(
            user_query="Tell me about SQURPHARMA",
            user_profile="moderate",
            capital_bdt=150_000.0,
        ),
    ]

    for q in queries:
        print(f"\n>>> USER: {q.user_query}")
        try:
            result = orch.advise(q)
            print(fmt_result(result))
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n\n=== RANK-ALL: top 10 by ensemble signal (moderate, 100k) ===")
    try:
        df = orch.rank_all(profile="moderate", capital_bdt=100_000.0, top_k=10)
        cols = ["stock", "ensemble_signal", "confidence", "direction",
                "position_pct", "position_bdt"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
