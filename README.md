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
```

---

## 📊 Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Research Foundation | ✅ Complete |
| 1 | Data Engineering (30 stocks + 3 indices) | ✅ Complete |
| 2 | Data Processing (34 features) | ✅ Complete |
| 3 | Baseline ML (LR + RF + XGBoost) | ✅ Complete |
| 4 | Deep Learning (LSTM/GRU) | 📝 Pending |
| 5 | Transformers (Informer, etc.) | 📝 Pending |
| 6 | Sentiment Analysis | 📝 Pending |
| 7 | Multimodal Forecasting | 📝 Pending |
| 8 | Explainable AI (SHAP/LIME) | 📝 Pending |
| 9 | RAG System | 📝 Pending |
| 10 | Multi-Agent System | 📝 Pending |
| 11 | Portfolio Optimization | 📝 Pending |
| 12 | LLM Financial Advisor | 📝 Pending |
| 13 | Dashboard | 📝 Pending |
| 14 | Research Paper | 📝 Pending |
| 15 | Thesis Submission | 📝 Pending |

**Current Phase**: 3 (Baseline ML Complete)

**Latest Results** (30 stocks, 80/20 time-based split):
- Linear Regression: RMSE 16.47, R² 0.89
- Random Forest: RMSE 28.03, R² 0.21
- XGBoost: RMSE 29.43, R² 0.14

See `results/baseline/summary_report.txt` for full details.

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
