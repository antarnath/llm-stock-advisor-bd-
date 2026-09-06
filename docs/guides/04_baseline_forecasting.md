# PHASE 3 — Baseline Forecasting

**Duration**: 2 Weeks  
**Started**: Week 5  
**Status**: 📝 Pending  
**Goal**: Establish baseline benchmarks before deep learning

---

## 🎯 Objectives

1. Implement classical ML regression models
2. Train/test split (80/20 time-based)
3. Establish benchmark metrics
4. Compare models statistically
5. Create publication-quality comparison table

---

## 📊 Models to Implement

### **Model 1: Linear Regression**

**Purpose**: Simple, interpretable baseline

**Implementation**:
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)
```

**Pros**:
- Fast training
- Highly interpretable
- No hyperparameters
- Good baseline

**Cons**:
- Assumes linear relationships
- Cannot capture complex patterns
- Sensitive to outliers

---

### **Model 2: Random Forest**

**Purpose**: Ensemble method for non-linear patterns

**Implementation**:
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Feature importance
importances = model.feature_importances_
```

**Pros**:
- Handles non-linearity
- Built-in feature importance
- Robust to outliers
- No scaling required

**Cons**:
- Can overfit with deep trees
- Less interpretable
- Slower inference
- Larger model size

---

### **Model 3: XGBoost**

**Purpose**: High-performance gradient boosting

**Implementation**:
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)
```

**Pros**:
- State-of-the-art performance
- Built-in regularization
- Handles missing values
- GPU support

**Cons**:
- Many hyperparameters
- Can overfit
- Slower training
- Less interpretable

---

### **Model 4: LightGBM**

**Purpose**: Fast gradient boosting

**Implementation**:
```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)
```

**Pros**:
- Very fast training
- Memory efficient
- Handles large datasets
- Categorical features support

**Cons**:
- Can overfit small data
- Many hyperparameters
- Less mature than XGBoost

---

## 📏 Evaluation Metrics

### **Implementation**

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R²': r2_score(y_true, y_pred),
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }
    return metrics

def calculate_directional_accuracy(y_true, y_pred):
    # Check if predictions correctly identify price direction
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction) * 100
```

### **Metrics Explanation**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **RMSE** | √(Σ(y_pred - y_true)² / n) | Penalizes large errors |
| **MAE** | Σ\|y_pred - y_true\| / n | Average error magnitude |
| **MAPE** | Σ\|y_pred - y_true\| / \|y_true\| × 100 | Percentage error |
| **R²** | 1 - (SS_res / SS_tot) | Variance explained |
| **Dir. Acc** | % correct direction | Trading relevance |

---

## 🔬 Experimental Setup

### **Data Split**
```python
# Time-based split (no shuffle!)
split_date = '2023-01-01'

train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Validation set (last 20% of training)
val_split = int(len(train) * 0.8)
train_final = train[:val_split]
val = train[val_split:]
```

### **Cross-Validation**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Train and evaluate
```

### **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 8, 12],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
```

---

## 📂 Project Structure

```
models/
└── baseline/
    ├── linear_regression.py
    ├── random_forest.py
    ├── xgboost_model.py
    ├── lightgbm_model.py
    ├── evaluate.py
    ├── train_all.py
    └── results/
        ├── metrics.json
        ├── predictions/
        └── plots/
```

---

## 📊 Expected Results

### **Benchmark Table Template**

| Model | RMSE | MAE | MAPE (%) | R² | Dir. Acc (%) |
|-------|------|-----|----------|-----|--------------|
| Linear Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |

---

## 📈 Visualization

### **Prediction vs Actual Plots**
```python
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### **Residual Analysis**
```python
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals over time
    axes[0].plot(residuals)
    axes[0].set_title('Residuals Over Time')
    axes[0].axhline(y=0, color='r', linestyle='--')
    
    # Residual distribution
    axes[1].hist(residuals, bins=50)
    axes[1].set_title('Residual Distribution')
    
    plt.show()
```

---

## 📂 Scripts to Create

- `models/baseline/linear_regression.py`
- `models/baseline/random_forest.py`
- `models/baseline/xgboost_model.py`
- `models/baseline/lightgbm_model.py`
- `models/baseline/evaluate.py`
- `models/baseline/train_all.py`
- `models/baseline/visualize_results.py`

---

## ✅ Success Criteria

- [ ] All 4 baseline models implemented
- [ ] Models trained on full dataset (30 stocks)
- [ ] Evaluation metrics calculated
- [ ] Time-series cross-validation performed
- [ ] Hyperparameter tuning completed
- [ ] Comparison table generated
- [ ] Statistical significance tests done
- [ ] Visualizations created
- [ ] Results documented

---

## 📊 Statistical Tests

### **Diebold-Mariano Test**
```python
from diebold_mariano import dm_test

# Compare two models' forecasts
dm_stat, p_value = dm_test(
    y_true,
    pred_model_1,
    pred_model_2,
    h=1,  # forecast horizon
    loss='MSE'
)
```

### **Paired t-test**
```python
from scipy.stats import ttest_rel

# Compare R² scores across stocks
t_stat, p_value = ttest_rel(
    model_1_scores,
    model_2_scores
)
```

---

## 📝 Documentation

### **Results Report**
- Performance per stock
- Average performance
- Best/worst performing stocks
- Statistical significance
- Computational cost analysis

---

## 🛠️ Tools & Libraries

- **scikit-learn**: Linear Regression, Random Forest
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **SciPy**: Statistical tests

---

**Next Phase**: Phase 4 — Deep Learning Forecasting

**Last Updated**: 2026-08-13
