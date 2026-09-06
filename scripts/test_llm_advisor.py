"""
Test the LLM-orchestrated advisor end-to-end.

Runs 5 sample questions through src.orchestrator.Orchestrator.answer()
and verifies:
  - Ticker resolution
  - Evidence pack generation
  - Tool dispatch (RAG, agents)
  - Fallback template works when no LLM configured
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


print("=== Importing src.orchestrator ===")
from src.orchestrator import Orchestrator, ChatAnswer
check("Orchestrator imported", Orchestrator is not None)
check("ChatAnswer imported", ChatAnswer is not None)


print("\n=== Creating orchestrator (stub LLM) ===")
orch = Orchestrator()
check("orchestrator created", orch is not None)
check("agents wired", orch.agents is not None)
check("LLM client wired", orch.llm is not None)
check("RAG retriever loaded", orch._rag is not None)
check("ToolRegistry created", orch.tools is not None)


print("\n=== Test 1: ticker resolution + evidence ===")
r = orch.chat("Should I buy BATBC for the long term?")
check("Returns ChatAnswer", isinstance(r, ChatAnswer))
check("Ticker resolved to BATBC", r.ticker == "BATBC", f"got {r.ticker}")
check("Tools called (multi_agent + rag)", len(r.tools_called) >= 1,
      f"tools: {r.tools_called}")
check("Reply is non-empty", len(r.reply) > 50)
check("Reply contains BATBC", "BATBC" in r.reply)
check("Mode is template (no LLM)", r.mode == "template")


print("\n=== Test 2: company name alias ===")
r = orch.chat("How is Grameenphone doing?")
check("Alias resolved to GP", r.ticker == "GP", f"got {r.ticker}")
check("Reply mentions GP", "GP" in r.reply)


print("\n=== Test 3: Bangla query ===")
r = orch.chat("ব্যাংক ঋণ জালিয়াতি সম্পর্কে কী বলতে পারেন?")
check("Reply non-empty (Bangla handled)", len(r.reply) > 0)
print(f"  Preview: {r.reply[:120]}...")


print("\n=== Test 4: unknown stock ===")
r = orch.chat("What is the meaning of life?")
check("No ticker resolved", r.ticker is None)
check("Reply explains the limitation", "couldn't" in r.reply.lower() or "stock" in r.reply.lower())


print("\n=== Test 5: profile + capital in query ===")
# Direct use of the inner advise() through a manual query
from src.agents import AdvisorQuery
advisor_q = AdvisorQuery(user_query="aggressive trader with 1 lakh on BEXIMCO",
                          user_profile="aggressive", capital_bdt=100_000.0)
advisor_q = orch.agents.parse_query(advisor_q)
check("Aggressive profile parsed", advisor_q.user_profile == "aggressive")
check("1 lakh → 100,000", advisor_q.capital_bdt == 100_000.0)


print("\n=== Test 6: tool dispatch ===")
result = orch.tools.call("run_technical_agent", {"ticker": "GP"})
check("Technical tool returns string", isinstance(result, str))
check("Contains signal=", "signal=" in result)

result = orch.tools.call("rag_search", {"query": "BEXIMCO profit", "top_k": 3})
check("RAG tool returns string", isinstance(result, str))

result = orch.tools.call("get_latest_prediction", {"ticker": "GP"})
check("Prediction tool returns string", isinstance(result, str))


print("\n=== Test 7: LLM stub fallback ===")
from src.llm import LLMClient
stub = LLMClient(provider="stub")
check("Stub provider set", stub.provider == "stub")
check("Stub is not live", stub.is_live is False)

messages = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="Hello"),
] if False else None  # placeholder
# Use the real path
from src.llm import ChatMessage as CM2
resp = stub.chat([CM2(role="user", content="Hello")])
check("Stub returns content", len(resp.content) > 0)
check("Stub content mentions stub mode", "stub" in resp.content.lower())


print("\n=== Test 8: ticker resolver edge cases ===")
from src.orchestrator.ticker_resolver import resolve_ticker
universe = ["ACI", "BANKASIA", "BATBC", "BEXIMCO", "BEXPHARMA", "BRACBANK",
            "GP", "SQURPHARMA", "ISLAMI BANK", "RENATA"]
check("Direct ticker", resolve_ticker("How is BATBC?", universe) == "BATBC")
check("Company name alias", resolve_ticker("What about Grameenphone?", universe) == "GP")
check("Bank alias", resolve_ticker("Tell me about BRAC bank", universe) == "BRACBANK")
check("ISLAMI BANK wins over BANK", resolve_ticker("ISLAMI BANK news", universe) == "ISLAMI BANK")
check("No match", resolve_ticker("What's the weather?", universe) is None)


print("\n=== Summary ===")
print(f"  {passed} passed, {failed} failed")
if failed:
    sys.exit(1)