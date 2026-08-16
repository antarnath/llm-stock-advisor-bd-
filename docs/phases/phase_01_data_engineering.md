# PHASE 1 — Data Engineering

**Duration**: 2 Weeks  
**Started**: Week 2  
**Status**: 🔄 **IN PROGRESS**  
**Priority**: ⚠️ **CRITICAL - Most Important Phase**

This phase requires extensive time as data quality determines model performance.

---

## 🎯 Goals

1. Collect comprehensive historical stock data
2. Gather market indices
3. Compile company fundamentals
4. Scrape DSE announcements
5. Collect financial news
6. Download annual reports

---

## 📊 Dataset 1: Historical Stock Data

### **Schema**
```csv
date, code, open, high, low, close, volume, trade, value
```

### **Target Specifications**
- **Time Period**: 2010-2025 (15 years)
- **Coverage**: 100+ stocks
- **Frequency**: Daily trading data
- **Status**: ✅ **30 stocks collected**

### **Sample Data**
```
date,code,name,sector,open,high,low,close,volume,trade,value
2010-01-01,GP,Grameenphone Ltd,Telecom,396.82,402.22,383.29,391.79,334697,671,131129856.1
```

### **Data Sources**
- DSE official website
- Kaggle datasets
- GitHub repositories
- Bloomberg/Reuters (if available)

### **Top 30 Stocks Collected** ✅

| # | Code | Company | Sector |
|---|------|---------|--------|
| 1 | GP | Grameenphone Ltd | Telecom |
| 2 | BATBC | British American Tobacco Bangladesh | Tobacco |
| 3 | SQURPHARMA | Square Pharmaceuticals | Pharma |
| 4 | BRACBANK | BRAC Bank Ltd | Bank |
| 5 | WALTONHIL | Walton Hi-Tech Industries | Electronics |
| 6 | RENATA | Renata Ltd | Pharma |
| 7 | BEXIMCO | Beximco Ltd | Conglomerate |
| 8 | ISLAMI BANK | Islami Bank Bangladesh | Bank |
| 9 | DBBL | Dutch-Bangla Bank | Bank |
| 10 | DSEX | DSE Broad Index | Index |
| 11 | POWERGRID | Power Grid Company | Power |
| 12 | TITASGAS | Titas Gas | Gas |
| 13 | SUMITPOWER | Summit Power | Power |
| 14 | JAMUNAOIL | Jamuna Oil Company | Fuel |
| 15 | BANKASIA | Bank Asia Ltd | Bank |
| 16 | EBL | Eastern Bank Ltd | Bank |
| 17 | DUTCHBANGL | Dutch-Bangla Bank | Bank |
| 18 | BSCCL | Bangladesh Submarine Cable | Telecom |
| 19 | ROBI | Robi Axiata | Telecom |
| 20 | ACI | Advanced Chemical Industries | Pharma |
| 21 | BEXPHARMA | Beximco Pharmaceuticals | Pharma |
| 22 | MARICO | Marico Bangladesh | Consumer |
| 23 | UNILEVER | Unilever Bangladesh | Consumer |
| 24 | HEIDELBCEM | Heidelberg Cement | Cement |
| 25 | LAFARGECEM | LafargeHolcim Bangladesh | Cement |
| 26 | MARICO | Marico Bangladesh | Consumer |
| 27 | MUTUALTRUST | Mutual Trust Bank | Bank |
| 28 | NCCBANK | NCC Bank | Bank |
| 29 | PRIMEBANK | Prime Bank | Bank |
| 30 | SIBL | Social Islami Bank | Bank |

---

## 📊 Dataset 2: Market Indices

### **Indices to Collect**
- **DSEX**: DSE Broad Index (primary benchmark)
- **DS30**: Top 30 companies index
- **DSES**: Shariah-compliant index

### **Format**
```csv
date, index_name, value, change, volume
```

### **Tasks**
- [ ] Collect DSEX daily data (2010-2025)
- [ ] Collect DS30 daily data (2010-2025)
- [ ] Collect DSES daily data (2010-2025)
- [ ] Save to `dataset/index/`

---

## 📊 Dataset 3: Company Fundamentals

### **Metrics to Collect**
- EPS (Earnings Per Share)
- NAV (Net Asset Value)
- PE Ratio (Price-to-Earnings)
- Dividend Yield
- Revenue
- Profit
- Market Cap
- Book Value

### **Format**
```csv
date, code, eps, nav, pe_ratio, dividend, revenue, profit
```

### **Tasks**
- [ ] Collect quarterly fundamentals for all 30 stocks
- [ ] Save to `dataset/fundamentals/`

---

## 📊 Dataset 4: DSE Announcements

### **Categories**
- Dividend declarations
- Board meeting notices
- Rights issues
- AGM (Annual General Meeting) notices
- Quarterly reports
- Stock splits
- Bonus shares

### **Format**
```csv
date, company, category, announcement_text, link
```

### **Tasks**
- [ ] Scrape DSE announcements (2010-2025)
- [ ] Categorize by type
- [ ] Save to `dataset/announcements/`

---

## 📊 Dataset 5: Financial News

### **Schema**
```csv
date, headline, content, company, source, sentiment
```

### **Sources**
- The Daily Star (Business)
- Dhaka Tribune
- Reuters Bangladesh
- Bloomberg South Asia
- BD News Today

### **Tasks**
- [ ] Scrape news articles (2010-2025)
- [ ] Extract company mentions
- [ ] Save to `dataset/news/`

---

## 📊 Dataset 6: Annual Reports

### **Format**: PDF documents

### **Structure**
```
annual_reports/
├── GP/
│   ├── 2020.pdf
│   ├── 2021.pdf
│   └── 2022.pdf
├── BATBC/
│   └── ...
```

### **Tasks**
- [ ] Download annual reports for all 30 stocks
- [ ] Organize by company and year
- [ ] Save to `dataset/annual_reports/`

---

## 📂 Final Deliverables Structure

```
dataset/
├── historical/          # ✅ 30 stocks ready
│   ├── GP.csv
│   ├── BATBC.csv
│   └── ...
├── index/              # 📝 DSEX, DS30, DSES
├── fundamentals/       # 📝 EPS, NAV, PE, etc.
├── announcements/      # 📝 DSE announcements
├── news/              # 📝 Financial news
├── annual_reports/    # 📝 PDF reports
└── processed/         # 📝 Phase 2
```

---

## 🛠️ Scripts Created

### **✅ collect_top_stocks.py**
- Location: `scripts/collect_top_stocks.py`
- Purpose: Collect top 30 stocks historical data
- Status: ✅ Working

### **📝 To Create**
- `collect_index.py` - Market indices
- `collect_fundamentals.py` - Company fundamentals
- `collect_announcements.py` - DSE announcements
- `collect_news.py` - Financial news
- `download_reports.py` - Annual reports

---

## 📊 Statistics

- **Total CSV files**: 30 (✅)
- **Total data points**: 125,220
- **Date range**: 2010-01-01 to 2025-12-31
- **Trading days per stock**: 4,174
- **Total file size**: 12 MB

---

## ✅ Success Criteria

- [x] 30 stocks historical data collected
- [ ] Market indices (DSEX, DS30, DSES) collected
- [ ] Company fundamentals collected
- [ ] DSE announcements scraped
- [ ] Financial news collected
- [ ] Annual reports downloaded
- [ ] All data saved in CSV/structured format
- [ ] Data validated and quality checked

---

## 🔄 Current Status

**Completed**: 30% (Historical stock data only)
- ✅ Dataset 1: Historical Stock Data (30 stocks)

**Pending**: 70%
- [ ] Dataset 2: Market Indices
- [ ] Dataset 3: Company Fundamentals
- [ ] Dataset 4: DSE Announcements
- [ ] Dataset 5: Financial News
- [ ] Dataset 6: Annual Reports

---

**Next Phase**: Phase 2 — Data Processing

**Last Updated**: 2026-08-13
