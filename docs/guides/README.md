# 📋 Project Guides

Complete build guide for the **LLM-Orchestrated Financial Advisor for Bangladesh Stock Market** thesis project.

---

## 🎯 Quick Navigation

This folder contains detailed guides for each major capability in the project. Work through them in the suggested order below.

---

## 📚 All Guides (in recommended build order)

| # | Guide | Status | Purpose |
|---|-------|--------|---------|
| 1 | [research_foundation.md](research_foundation.md) | ✅ | Literature review, problem statement, research gap |
| 2 | [data_engineering.md](data_engineering.md) | ✅ | Collect 30 stocks, indices, news, fundamentals |
| 3 | [data_processing.md](data_processing.md) | ✅ | Clean, feature-engineer (27 technical indicators) |
| 4 | [baseline_forecasting.md](baseline_forecasting.md) | ✅ | Linear Regression, Random Forest, XGBoost |
| 5 | [deep_learning_forecasting.md](deep_learning_forecasting.md) | ✅ | LSTM, GRU, CNN-LSTM |
| 6 | [advanced_timeseries.md](advanced_timeseries.md) | ✅ | Informer, Autoformer, PatchTST transformers |
| 7 | [sentiment_analysis.md](sentiment_analysis.md) | ✅ | FinBERT + BanglaBERT on news articles |
| 8 | [multimodal_forecasting.md](multimodal_forecasting.md) | ✅ | Price + sentiment fusion (3 strategies) |
| 9 | [explainable_ai.md](explainable_ai.md) | ✅ | SHAP + LIME explanations for every prediction |
| 10 | [rag_system.md](rag_system.md) | ✅ | FAISS index over 1,560 multilingual news |
| 11 | [multi_agent_system.md](multi_agent_system.md) | 🔄 In progress | 4 specialist agents + orchestrator |
| 12 | [portfolio_optimization.md](portfolio_optimization.md) | 📝 Pending | Markowitz, Sharpe, risk parity |
| 13 | [llm_financial_advisor.md](llm_financial_advisor.md) | 📝 Pending | Chat interface tying everything together |
| 14 | [dashboard.md](dashboard.md) | 📝 Pending | Streamlit / Next.js UI |
| 15 | [research_paper.md](research_paper.md) | 📝 Pending | Publication-quality paper |
| 16 | [thesis_submission.md](thesis_submission.md) | 📝 Pending | Final thesis submission package |

**Supplementary material:**
- [gru_cnnlstm_extra.md](gru_cnnlstm_extra.md) — GRU/CNN-LSTM supplementary details
- [fingpt_banglabert_extra.md](fingpt_banglabert_extra.md) — FinGPT and BanglaBERT notes
- [advanced_multimodal_extra.md](advanced_multimodal_extra.md) — Tensor fusion experiments

---

## 🎯 Research Contributions

1. **Leak-Free DSE Benchmark** — First systematic comparison of 9+ models on Bangladesh data with strict time-based splits
2. **Bilingual Multimodal Forecasting** — Novel combination of Bangla + English news with price data for an emerging market
3. **LLM-Orchestrated Advisor** — Multi-agent framework with explainable reasoning
4. **Open Reproducible Dataset** — Curated DSE-BD released on HuggingFace

---

## 🛠️ Suggested Build Order

1. **Foundation** — Data → Features → Models (research → engineering → ML)
2. **Enhancement** — Sentiment → Multimodal → XAI
3. **Knowledge** — RAG over news + explanations
4. **Intelligence** — Multi-agent orchestration + portfolio
5. **Product** — Dashboard + paper + thesis submission

---

## 📊 Status Legend

- ✅ **Completed** — finished and integrated
- 🔄 **In Progress** — currently being built
- 📝 **Pending** — not yet started
- ⚠️ **Critical** — most important capability

---

## 💡 How to Use This Folder

1. Each guide contains: objectives, code patterns, success criteria, tools
2. Read the next pending guide to know what to build next
3. The folder name describes the capability, not a fixed order number

---

## 🎓 Final Goal

Complete a research-grade thesis with:
- ✅ Publication-quality paper
- ✅ Production-ready system
- ✅ Open-source contribution
- ✅ Novel research contributions
- ✅ Practical impact for Bangladesh

---

**Last Updated**: 2026-09-06
