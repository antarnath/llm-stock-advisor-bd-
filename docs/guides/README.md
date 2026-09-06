# Project Guides

Complete build guide for the **LLM-Orchestrated Financial Advisor for Bangladesh Stock Market** thesis project.

---

## Quick Navigation

This folder contains detailed guides for each major capability in the project. Work through them in the suggested order below.

---

## All Guides (in build order)

| # | Guide | Status | Purpose |
|---|-------|--------|---------|
| 01 | [Research Foundation](01_research_foundation.md) | ✅ | Literature review, problem statement, research gap |
| 02 | [Data Engineering](02_data_engineering.md) | ✅ | Collect 30 stocks, indices, news, fundamentals |
| 03 | [Data Processing](03_data_processing.md) | ✅ | Clean, feature-engineer (27 technical indicators) |
| 04 | [Baseline Forecasting](04_baseline_forecasting.md) | ✅ | Linear Regression, Random Forest, XGBoost |
| 05 | [Baseline Models](05_baseline_models.md) | ✅ | Consolidated baseline model reference |
| 06 | [Deep Learning Forecasting](06_deep_learning_forecasting.md) | ✅ | LSTM (leak-free, time-based splits) |
| 07 | [Advanced Time-Series](07_advanced_timeseries.md) | ✅ | Informer, Autoformer, PatchTST transformers |
| 08 | [Sentiment Analysis](08_sentiment_analysis.md) | ✅ | FinBERT + BanglaBERT on 1,560 news articles |
| 09 | [Multimodal Forecasting](09_multimodal_forecasting.md) | ✅ | Price + sentiment fusion (early / late / attention) |
| 10 | [Explainable AI](10_explainable_ai.md) | ✅ | SHAP + LIME on all 30 stocks, NL explanations |
| 11 | [RAG System](11_rag_system.md) | ✅ | FAISS index, multilingual query (en + bn) |
| 12 | [Multi-Agent System](12_multi_agent_system.md) | ✅ | 4 specialist agents + orchestrator |
| 13 | [Portfolio Optimization](13_portfolio_optimization.md) | ⏸ Partial | Rule-based sizing done; full MPT deferred |
| 14 | [LLM Financial Advisor](14_llm_financial_advisor.md) | ⏸ Partial | Chat UI scaffolded; LLM integration pending |
| 15 | [Dashboard](15_dashboard.md) | ✅ | Next.js 14 frontend (chat, ticker, settings) |
| 16 | [Research Paper](16_research_paper.md) | ⏸ Partial | Uniqueness analysis + ablation done; draft pending |
| 17 | [Thesis Submission](17_thesis_submission.md) | 📝 Pending | Final compilation & submission |
| 18 | [GRU + CNN-LSTM Extra](18_gru_cnnlstm_extra.md) | 📝 Deferred | Supplementary DL architectures |
| 19 | [FinGPT + BanglaBERT Extra](19_fingpt_banglabert_extra.md) | 📝 Deferred | GPU-only advanced sentiment |
| 20 | [Advanced Multimodal Extra](20_advanced_multimodal_extra.md) | 📝 Deferred | Macro + per-event fusion experiments |

---

## Research Contributions

1. **Leak-Free DSE Benchmark** — First systematic comparison of 9+ models on Bangladesh data with strict time-based splits.
2. **Bilingual Multimodal Forecasting** — Novel combination of Bangla + English news with price data for an emerging market.
3. **LLM-Orchestrated Advisor** — Multi-agent framework with explainable reasoning.
4. **Open Reproducible Dataset** — Curated DSE-BD released on HuggingFace.

---

## Suggested Build Order

1. **Foundation** — Data → Features → Models (research → engineering → ML)
2. **Enhancement** — Sentiment → Multimodal → XAI
3. **Knowledge** — RAG over news + explanations
4. **Intelligence** — Multi-agent orchestration + portfolio
5. **Product** — Dashboard + paper + thesis submission

---

## Status Legend

- ✅ **Completed** — finished and integrated
- ⏸ **Partial** — some components shipped; full scope pending
- 📝 **Pending** — not yet started
- 📝 **Deferred** — explicitly out of scope for the current milestone

---

## How to Use This Folder

1. Each guide contains: objectives, code patterns, success criteria, tools.
2. Open the next pending guide to know what to build next.
3. Filename numbers are stable order references; status emojis show progress.

---

## Final Goal

Complete a research-grade thesis with:
- ✅ Publication-quality paper (in progress)
- ✅ Production-ready system
- ✅ Open-source contribution
- ✅ Novel research contributions
- ✅ Practical impact for Bangladesh

---

**Last Updated**: 2026-09-06
