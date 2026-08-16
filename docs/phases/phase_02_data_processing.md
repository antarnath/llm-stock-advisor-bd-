# PHASE 2 — Data Processing

**Duration**: 1 Week  
**Started**: Week 4  
**Status**: 📝 Pending  
**Goal**: Transform raw data into clean, feature-rich datasets

---

## 🎯 Objectives

1. Clean raw data
2. Handle missing values
3. Remove duplicates
4. Detect outliers
5. Engineer technical features
6. Store in PostgreSQL

---

## 🧹 Data Cleaning

### **Missing Values**

**Strategy**:
- Forward fill for small gaps (< 3 days)
- Linear interpolation for medium gaps (3-7 days)
- Remove stocks with >30% missing data
- Mark holidays/exchange closures

**Implementation**:
```python
# Forward fill
df.fillna(method='ffill', limit=2, inplace=True)

# Interpolation
df.interpolate(method='linear', inplace=True)

# Remove high-missing stocks
missing_pct = df.isnull().sum() / len(df)
if missing_pct > 0.3:
    df.dropna(thresh=len(df)*0.7, axis=1, inplace=True)
```

### **Duplicate Removal**

**Strategy**:
- Identify exact duplicate dates
- Verify with multiple sources
- Keep latest entry
- Log duplicates removed

**Implementation**:
```python
# Check duplicates
duplicates = df[df.duplicated(subset=['date'], keep=False)]
print(f"Found {len(duplicates)} duplicate dates")

# Remove duplicates
df.drop_duplicates(subset=['date'], keep='last', inplace=True)
```

### **Outlier Detection**

**Methods**:
1. **Z-score Method**: |z| > 3
2. **IQR Method**: 1.5 × IQR
3. **Domain Knowledge**: Verify extreme events

**Implementation**:
```python
# Z-score method
from scipy import stats
z_scores = stats.zscore(df['close'])
outliers = df[abs(z_scores) > 3]

# IQR method
Q1 = df['close'].quantile(0.25)
Q3 = df['close'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['close'] < Q1 - 1.5*IQR) | (df['close'] > Q3 + 1.5*IQR)]
```

---

## 🔧 Feature Engineering

### **1. SMA - Simple Moving Average**

```python
# 20-day SMA
df['SMA_20'] = df['close'].rolling(window=20).mean()

# Multiple windows
for window in [5, 10, 20, 50, 100, 200]:
    df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
```

### **2. EMA - Exponential Moving Average**

```python
# EMA formula: EMA(t) = α × price(t) + (1-α) × EMA(t-1)
def calculate_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()

for span in [12, 26, 50]:
    df[f'EMA_{span}'] = calculate_ema(df['close'], span)
```

### **3. RSI - Relative Strength Index**

```python
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['close'], 14)
```

### **4. MACD - Moving Average Convergence Divergence**

```python
df['EMA_12'] = calculate_ema(df['close'], 12)
df['EMA_26'] = calculate_ema(df['close'], 26)
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = calculate_ema(df['MACD'], 9)
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
```

### **5. Bollinger Bands**

```python
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['BB_Std'] = df['close'].rolling(window=20).std()
df['BB_Upper'] = df['SMA_20'] + (2 * df['BB_Std'])
df['BB_Lower'] = df['SMA_20'] - (2 * df['BB_Std'])
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
df['BB_Position'] = (df['close'] - df['BB_Lower']) / df['BB_Width']
```

### **6. ATR - Average True Range**

```python
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

df['ATR_14'] = calculate_atr(df, 14)
```

### **7. Volatility**

```python
# Daily returns
df['Returns'] = df['close'].pct_change()

# Rolling volatility (annualized)
df['Volatility_30'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
df['Volatility_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
```

### **8. Returns**

```python
# Simple returns
df['Returns_1d'] = df['close'].pct_change(1)
df['Returns_5d'] = df['close'].pct_change(5)
df['Returns_20d'] = df['close'].pct_change(20)

# Log returns
df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
```

---

## 💾 Database Storage

### **PostgreSQL Schema**

```sql
-- Stocks table
CREATE TABLE stocks (
    code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200),
    sector VARCHAR(100),
    listing_date DATE,
    market_cap BIGINT
);

-- Prices table
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) REFERENCES stocks(code),
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    trade INTEGER,
    value DECIMAL(15,2)
);

-- Fundamentals table
CREATE TABLE fundamentals (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) REFERENCES stocks(code),
    date DATE,
    eps DECIMAL(10,2),
    nav DECIMAL(10,2),
    pe_ratio DECIMAL(10,2),
    dividend DECIMAL(10,2),
    revenue BIGINT,
    profit BIGINT
);

-- News table
CREATE TABLE news (
    id SERIAL PRIMARY KEY,
    date DATE,
    headline TEXT,
    content TEXT,
    company VARCHAR(20),
    source VARCHAR(100),
    sentiment VARCHAR(20)
);

-- Reports table
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) REFERENCES stocks(code),
    year INTEGER,
    report_type VARCHAR(50),
    file_path VARCHAR(500)
);
```

### **Setup Script**

```python
# scripts/setup_database.py
import psycopg2
from sqlalchemy import create_engine

def setup_database():
    # Connection
    conn = psycopg2.connect(
        host="localhost",
        database="dse_research",
        user="researcher",
        password="password"
    )
    
    cur = conn.cursor()
    
    # Create tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code VARCHAR(20) PRIMARY KEY,
            name VARCHAR(200),
            sector VARCHAR(100)
        )
    """)
    
    # ... (other tables)
    
    conn.commit()
    cur.close()
    conn.close()
```

---

## 📂 Deliverables

```
processed/
├── features/          # Engineered features (CSV per stock)
├── train_test_splits/ # Train/test datasets
├── visualizations/    # Data exploration plots
└── database/         # PostgreSQL dump files
```

### **Files to Create**
- `scripts/data_cleaning.py` - Clean raw data
- `scripts/feature_engineering.py` - Create features
- `scripts/setup_database.py` - Setup PostgreSQL
- `scripts/load_to_database.py` - Load data to DB

---

## ✅ Success Criteria

- [ ] All missing values handled
- [ ] Duplicates removed
- [ ] Outliers detected and handled
- [ ] Technical indicators calculated (SMA, EMA, RSI, MACD, BB, ATR)
- [ ] Volatility and returns calculated
- [ ] PostgreSQL database setup
- [ ] Data loaded to database
- [ ] Train/test splits created (80/20)
- [ ] Visualizations generated

---

## 📊 Data Quality Metrics

- **Completeness**: % of non-null values
- **Consistency**: Duplicate dates count
- **Accuracy**: Outlier percentage
- **Timeliness**: Date range coverage

---

## 🛠️ Tools

- **Python**: Pandas, NumPy, SciPy
- **Database**: PostgreSQL, SQLAlchemy
- **Visualization**: Matplotlib, Seaborn, Plotly

---

**Next Phase**: Phase 3 — Baseline Forecasting

**Last Updated**: 2026-08-13