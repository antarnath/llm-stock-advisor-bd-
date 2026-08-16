# LLM-Orchestrated Financial Advisor for Bangladesh Stock Market

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/status-active-green.svg)]()

A research-grade thesis project implementing an **LLM-orchestrated financial advisor** for the **Dhaka Stock Exchange (DSE)**. The system combines statistical ML, deep learning, transformer-based time-series models, sentiment analysis, RAG, and multi-agent LLM orchestration to provide actionable financial advice for Bangladeshi retail investors.

---

## 🎯 Project Goals

1. **Forecast** DSE stock prices with high accuracy using ML/DL/Transformer models
2. **Analyze sentiment** from financial news in Bangla and English
3. **Build a RAG system** over DSE announcements, annual reports, and news
4. **Orchestrate 6 specialized LLM agents** for end-to-end financial advice
5. **Optimize portfolios** using Modern Portfolio Theory
6. **Deploy** a production-grade dashboard and REST API

---

## 📂 Project Structure

```
Dataset/
├── README.md                  # ← you are here
├── requirements.txt           # Python dependencies
├── .env.example               # API keys template
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── raw/                   # Original scraped data
│   │   ├── stocks/            #   30 DSE stocks (2010-2026)
│   │   └── indices/           #   DSEX, DS30, DSES market indices
│   ├── processed/             # Phase 2: + technical indicators
│   └── external/              # Fundamentals, news (Phase 6+)
│
├── models/
│   ├── baseline/              # Phase 3: Linear Reg, RF, XGBoost
│   ├── deep_learning/         # Phase 4: LSTM, GRU, CNN-LSTM
│   ├── transformers/          # Phase 5: Informer, Autoformer, PatchTST
│   └── experiments/           # Phase 7: multimodal models
│
├── src/                       # Production source code
│   ├── data_collection/       #   Scrape stocks/indices from DSE
│   ├── data_processing/       #   Technical indicators, features
│   ├── training/              #   Model training pipelines
│   ├── evaluation/            #   Metrics, visualizations, reports
│   ├── inference/             #   Load models, predict
│   └── utils/                 #   config, logger
│
├── scripts/
│   ├── debug/                 #   Ad-hoc debugging scripts
│   └── run_pipeline.py        #   End-to-end pipeline runner
│
├── docs/                      # All documentation
│   ├── phases/                #   16 phase specifications
│   ├── PROGRESS.md            #   Current progress
│   ├── architecture/          #   System architecture diagrams
│   └── api/                   #   API documentation
│
├── notebooks/                 # Jupyter exploration
├── tests/                     # Unit tests
├── logs/                      # Runtime logs
└── results/                   # Output artifacts
    └── baseline/              #   Phase 3 baseline results + plots
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd /media/antar-chandra-nath/Media/Research/Dataset

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Collect DSE stock data (Phase 1)
2. Process and add technical indicators (Phase 2)
3. Train baseline models (Phase 3)
4. Generate predictions (Phase 3)
5. Create visualizations (Phase 3)

### 3. Run Individual Phases

```bash
# Phase 1: Collect stock data
python src/data_collection/collect_stocks.py

# Phase 2: Process and add indicators
python src/data_processing/technical_indicators.py

# Phase 3: Train baseline models
python src/training/baseline_trainer.py

# Phase 3: Generate visualizations
python src/evaluation/visualize.py

# Phase 3: Make predictions
python src/inference/predict.py

# Phase 4: Deep Learning (LSTM)
python src/training/deep_learning_trainer.py

# Phase 6: Sentiment Analysis
python src/data_collection/news_curator.py            # Generate 1,560 labelled articles
python src/sentiment/scoring_pipeline.py              # Score with FinBERT/BanglaLexicon
python src/sentiment/correlation_analysis.py          # Sentiment-return correlations
python src/sentiment/visualize.py                     # 7 plots + summary report

# Phase 7: Multimodal Forecasting
python src/training/multimodal_trainer.py             # Train early+late fusion on 30 stocks
python src/inference/mm_predict.py                    # 5-day predictions (sentiment frozen)
python src/evaluation/mm_visualize.py                 # 8 plots + ablation vs Phase 4
```

---

## 📊 Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Research Foundation | ✅ Complete |
| 1 | Data Engineering (30 stocks + 3 indices) | ✅ Complete |
| 2 | Data Processing (34 features) | ✅ Complete |
| 3 | Baseline ML (LR + RF + XGBoost + LightGBM, leak-free v2) | ✅ Complete |
| 4 | Deep Learning (LSTM only — GRU + CNN-LSTM deferred to Phase 4xxx) | ✅ Complete |
| 5 | Transformers (Informer, etc.) | 📝 Deferred |
| 6 | Sentiment Analysis (FinBERT + VADER + BanglaLexicon) | ✅ Complete |
| 7 | Multimodal Forecasting (early + late fusion) | ✅ Complete |
| 8 | Explainable AI (SHAP/LIME) | 📝 Pending |
| 9 | RAG System | 📝 Pending |
| 10 | Multi-Agent System | 📝 Pending |
| 11 | Portfolio Optimization | 📝 Pending |
| 12 | LLM Financial Advisor | 📝 Pending |
| 13 | Dashboard | 📝 Pending |
| 14 | Research Paper | 📝 Pending |
| 15 | Thesis Submission | 📝 Pending |

**Current Phase**: 8 (Explainable AI — SHAP/LIME on the multimodal models)

**Latest Results** (30 stocks, 80/20 time-based split, leak-free v2):
- Linear Regression: RMSE 0.0198, R² -0.016, Dir_Acc 50.0%
- Random Forest: RMSE 0.0202, R² -0.055, Dir_Acc 50.2%
- XGBoost: RMSE 0.0202, R² -0.061, Dir_Acc 50.2%
- LightGBM: RMSE 0.0205, R² -0.088, Dir_Acc 50.2%
- LSTM (Phase 4): RMSE 0.0198, R² -0.01, Dir_Acc 49.8%
- Multimodal Early (Phase 7): see `results/multimodal/summary_report.txt`
- Multimodal Late (Phase 7): see `results/multimodal/summary_report.txt`

See `results/baseline/summary_report.txt` for Phase 3, `results/deep_learning/summary_report.txt` for Phase 4, `results/sentiment/summary_report.txt` for Phase 6, and `results/multimodal/summary_report.txt` for Phase 7.

**Phase 7 Multimodal Architecture**:
- **Early fusion**: Concat price (60×27) + sentiment (60×7) → single 3-layer LSTM (356K params)
- **Late fusion**: Separate price LSTM (128, 2-layer) + sentiment LSTM (32, 1-layer) → concat → MLP (228K params)
- **Sentiment policy**: Per-stock forward-fill (sticky) + bfill for leading NaNs + 0.0 fallback
- **Inference**: Sentiment window FROZEN at last-known across all forecast steps; only price iterates

---

## 🛠️ Tech Stack

**Languages:** Python 3.10+
**Core ML:** scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow
**Time Series:** Informer, Autoformer, PatchTST, NeuralForecast
**NLP:** Hugging Face Transformers, FinBERT, FinGPT, LangChain
**Visualization:** Matplotlib, Seaborn, Plotly
**Backend:** FastAPI, PostgreSQL, FAISS/ChromaDB
**Frontend:** Next.js, TypeScript, Tailwind CSS, Recharts
**Deployment:** Docker, AWS/GCP

---

## 📚 Documentation

- **Project phases:** [`docs/phases/`](docs/phases/) — 16 detailed phase specs
- **Progress tracking:** [`docs/PROGRESS.md`](docs/PROGRESS.md)
- **Architecture:** [`docs/architecture/`](docs/architecture/)
- **API docs:** [`docs/api/`](docs/api/)

---

## 🤝 Contributing

This is a thesis project by **Antar Chandra Nath**. Suggestions and feedback welcome via GitHub issues.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📞 Contact

**Author:** Antar Chandra Nath
**Project:** LLM-Orchestrated Financial Advisor for Bangladesh Stock Market
**Date:** 2026-08-13

---

*Built with ❤️ for the Bangladeshi financial research community*
