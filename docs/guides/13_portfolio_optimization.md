# PHASE 11 — Portfolio Optimization

**Duration**: 1 Week  
**Started**: Week 22  
**Status**: 📝 Pending  
**Goal**: Implement portfolio optimization algorithms

---

## 🎯 Objectives

1. Implement Modern Portfolio Theory (MPT)
2. Calculate efficient frontier
3. Optimize for Sharpe ratio
4. Add constraints and customization
5. Build backtesting framework

---

## 📊 Modern Portfolio Theory (MPT)

### **Markowitz Model**

The foundation of portfolio optimization:
- **Expected Return**: Weighted average of individual returns
- **Risk (Variance)**: Weighted covariance matrix
- **Objective**: Maximize return for given risk OR minimize risk for given return

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Modern Portfolio Theory implementation"""
    
    def __init__(self, returns_data):
        """
        Args:
            returns_data: DataFrame of stock returns (dates x stocks)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean() * 252  # Annualized
        self.cov_matrix = returns_data.cov() * 252  # Annualized
        self.n_stocks = len(returns_data.columns)
    
    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility"""
        weights = np.array(weights)
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights))
        )
        sharpe_ratio = (portfolio_return - 0.05) / portfolio_volatility  # Risk-free = 5%
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Negative Sharpe ratio (for minimization)"""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Portfolio volatility (for minimization)"""
        return self.portfolio_performance(weights)[1]
    
    def optimize_sharpe(self, constraints=None):
        """Maximize Sharpe ratio"""
        # Initial guess (equal weights)
        init_weights = np.array([1/self.n_stocks] * self.n_stocks)
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(self.n_stocks))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints:
            cons.extend(constraints)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return {
            'weights': result.x,
            'performance': self.portfolio_performance(result.x),
            'success': result.success
        }
    
    def optimize_min_volatility(self, target_return=None):
        """Minimize volatility for given return"""
        init_weights = np.array([1/self.n_stocks] * self.n_stocks)
        bounds = tuple((0, 1) for _ in range(self.n_stocks))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.mean_returns) - target_return
            })
        
        result = minimize(
            self.portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return {
            'weights': result.x,
            'performance': self.portfolio_performance(result.x),
            'success': result.success
        }
```

---

## 📈 Efficient Frontier

```python
class EfficientFrontier:
    """Calculate and visualize the efficient frontier"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def calculate_frontier(self, num_portfolios=100):
        """Calculate efficient frontier points"""
        # Get min and max returns
        min_ret = self.optimizer.mean_returns.min()
        max_ret = self.optimizer.mean_returns.max()
        
        # Target returns
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        frontier_volatility = []
        frontier_returns = []
        frontier_weights = []
        
        for target in target_returns:
            result = self.optimizer.optimize_min_volatility(target_return=target)
            if result['success']:
                frontier_volatility.append(result['performance'][1])
                frontier_returns.append(target)
                frontier_weights.append(result['weights'])
        
        return {
            'returns': frontier_returns,
            'volatility': frontier_volatility,
            'weights': frontier_weights
        }
    
    def plot_frontier(self, frontier_data, optimal_portfolio=None):
        """Visualize efficient frontier"""
        plt.figure(figsize=(12, 7))
        
        # Plot frontier
        plt.plot(
            frontier_data['volatility'], 
            frontier_data['returns'],
            'b--', linewidth=3, label='Efficient Frontier'
        )
        
        # Plot individual stocks
        for i, stock in enumerate(self.optimizer.returns.columns):
            plt.scatter(
                np.sqrt(self.optimizer.cov_matrix.iloc[i, i]),
                self.optimizer.mean_returns.iloc[i],
                s=100, label=stock
            )
        
        # Plot optimal portfolio
        if optimal_portfolio:
            plt.scatter(
                optimal_portfolio['performance'][1],
                optimal_portfolio['performance'][0],
                marker='*', s=500, c='red', label='Optimal Portfolio'
            )
        
        # Capital Market Line
        if optimal_portfolio:
            sharpe = optimal_portfolio['performance'][2]
            risk_free = 0.05
            cml_x = np.linspace(0, max(frontier_data['volatility']), 100)
            cml_y = risk_free + sharpe * cml_x
            plt.plot(cml_x, cml_y, 'g--', alpha=0.5, label='Capital Market Line')
        
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
```

---

## 🎯 Advanced Optimization

### **Risk Parity Portfolio**

```python
class RiskParityOptimizer:
    """Risk parity - equal risk contribution from each asset"""
    
    def __init__(self, returns_data):
        self.returns = returns_data
        self.cov_matrix = returns_data.cov() * 252
    
    def risk_contribution(self, weights):
        """Calculate risk contribution of each asset"""
        weights = np.array(weights)
        portfolio_vol = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights))
        )
        marginal_risk = np.dot(self.cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_risk
        return risk_contrib
    
    def risk_budget_objective(self, weights, target_risk_contrib):
        """Minimize deviation from target risk contribution"""
        risk_contrib = self.risk_contribution(weights)
        # Normalize
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        # Squared error
        return np.sum((risk_contrib_pct - target_risk_contrib) ** 2)
    
    def optimize(self):
        """Find risk parity portfolio"""
        n = len(self.cov_matrix)
        init_weights = np.array([1/n] * n)
        bounds = tuple((0.01, 0.5) for _ in range(n))  # Min 1%, max 50%
        
        # Equal risk contribution
        target_risk = np.array([1/n] * n)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            self.risk_budget_objective,
            init_weights,
            args=(target_risk,),
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return {
            'weights': result.x,
            'risk_contributions': self.risk_contribution(result.x),
            'success': result.success
        }
```

### **Black-Litterman Model**

```python
class BlackLitterman:
    """Black-Litterman model with investor views"""
    
    def __init__(self, returns_data, market_caps=None):
        self.returns = returns_data
        self.cov_matrix = returns_data.cov() * 252
        self.n = len(returns_data.columns)
        
        # Market cap weights (or equal if not provided)
        if market_caps is not None:
            self.market_weights = market_caps / market_caps.sum()
        else:
            self.market_weights = np.array([1/self.n] * self.n)
        
        # Implied equilibrium returns (reverse optimization)
        self.risk_aversion = 2.5  # Typical value
        self.pi = self.risk_aversion * np.dot(self.cov_matrix, self.market_weights)
    
    def optimize_with_views(self, views, view_confidences):
        """
        views: dict like {'GP': 0.15, 'BATBC': 0.08}
        view_confidences: dict with confidence values (0-1)
        """
        # Construct view matrix P and view vector Q
        P = np.zeros((len(views), self.n))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (stock, view_return) in enumerate(views.items()):
            stock_idx = list(self.returns.columns).index(stock)
            P[i, stock_idx] = 1
            Q[i] = view_return
            
            # Uncertainty in views (higher confidence = lower uncertainty)
            confidence = view_confidences.get(stock, 0.5)
            Omega[i, i] = (1 - confidence) * 0.01
        
        # Tau (scaling factor)
        tau = 0.05
        
        # Black-Litterman formula
        M_inv = np.linalg.inv(
            tau * self.cov_matrix + 
            np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        )
        
        bl_returns = np.dot(
            M_inv,
            np.dot(tau * self.cov_matrix, self.pi) + 
            np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        )
        
        # New covariance matrix
        bl_cov = self.cov_matrix + M_inv
        
        return {
            'expected_returns': bl_returns,
            'cov_matrix': bl_cov
        }
```

---

## 📊 Portfolio Constraints

```python
class ConstrainedOptimizer:
    """Portfolio optimization with custom constraints"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def add_sector_constraint(self, sector_mapping, max_sector_weight):
        """Add sector concentration limits"""
        constraints = []
        for sector, stocks in sector_mapping.items():
            indices = [list(self.optimizer.returns.columns).index(s) 
                      for s in stocks if s in self.optimizer.returns.columns]
            
            if indices:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=indices: max_sector_weight - sum(w[i] for i in idx)
                })
        return constraints
    
    def add_position_constraint(self, max_position=0.30):
        """Limit maximum position size"""
        constraints = []
        for i in range(self.optimizer.n_stocks):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=i: max_position - w[idx]
            })
        return constraints
    
    def add_minimum_position(self, min_position=0.05):
        """Ensure minimum diversification"""
        constraints = []
        for i in range(self.optimizer.n_stocks):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=i: w[idx] - min_position
            })
        return constraints
    
    def optimize_with_constraints(self, constraint_types):
        """Optimize with multiple constraints"""
        all_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if 'sector' in constraint_types:
            # Add sector constraints
            pass
        
        if 'max_position' in constraint_types:
            all_constraints.extend(self.add_position_constraint())
        
        if 'min_position' in constraint_types:
            all_constraints.extend(self.add_minimum_position())
        
        # Optimize
        result = self.optimizer.optimize_sharpe(constraints=all_constraints)
        return result
```

---

## 🔄 Backtesting Framework

```python
class PortfolioBacktester:
    """Backtest portfolio strategies"""
    
    def __init__(self, returns_data, prices_data):
        self.returns = returns_data
        self.prices = prices_data
    
    def backtest_equal_weight(self, rebalance_freq='monthly'):
        """Backtest equal-weight portfolio"""
        portfolio_value = 1.0
        portfolio_history = [portfolio_value]
        weights = np.array([1/len(self.returns.columns)] * len(self.returns.columns))
        
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)
        
        for i in range(1, len(self.returns)):
            # Calculate daily return
            daily_return = np.dot(weights, self.returns.iloc[i].values)
            portfolio_value *= (1 + daily_return)
            portfolio_history.append(portfolio_value)
            
            # Rebalance
            if self.returns.index[i] in rebalance_dates:
                weights = np.array([1/len(self.returns.columns)] * 
                                  len(self.returns.columns))
        
        return pd.Series(portfolio_history, index=self.returns.index)
    
    def backtest_dynamic(self, prediction_signals, rebalance_freq='weekly'):
        """Backtest using prediction signals"""
        portfolio_value = 1.0
        portfolio_history = [portfolio_value]
        
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)
        
        for i in range(1, len(self.returns)):
            # Use current weights
            if i == 1 or self.returns.index[i] in rebalance_dates:
                # Recalculate weights based on predictions
                weights = self.calculate_weights_from_signals(
                    prediction_signals.iloc[i]
                )
            
            daily_return = np.dot(weights, self.returns.iloc[i].values)
            portfolio_value *= (1 + daily_return)
            portfolio_history.append(portfolio_value)
        
        return pd.Series(portfolio_history, index=self.returns.index)
    
    def calculate_weights_from_signals(self, signals):
        """Convert prediction signals to portfolio weights"""
        # Long top predictions, short bottom (or zero for long-only)
        positive_signals = signals[signals > 0]
        
        if len(positive_signals) == 0:
            # Equal weight all
            return np.array([1/len(signals)] * len(signals))
        
        # Weight proportional to signal strength
        weights = positive_signals / positive_signals.sum()
        
        # Reindex to match all stocks
        full_weights = pd.Series(0, index=signals.index)
        full_weights.loc[weights.index] = weights
        
        return full_weights.values
    
    def calculate_metrics(self, portfolio_history):
        """Calculate performance metrics"""
        returns = portfolio_history.pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252 - 0.05) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(portfolio_history),
            'calmar_ratio': (returns.mean() * 252) / abs(self.calculate_max_drawdown(portfolio_history)),
            'win_rate': (returns > 0).sum() / len(returns),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }
        
        return metrics
    
    def calculate_max_drawdown(self, portfolio):
        """Calculate maximum drawdown"""
        cummax = portfolio.cummax()
        drawdown = (portfolio - cummax) / cummax
        return drawdown.min()
    
    def plot_performance(self, portfolio_history, benchmark=None):
        """Plot portfolio performance"""
        plt.figure(figsize=(14, 7))
        
        plt.plot(portfolio_history, label='Portfolio', linewidth=2)
        
        if benchmark is not None:
            plt.plot(benchmark, label='Benchmark (DSEX)', alpha=0.7)
        
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

---

## 🎨 Portfolio Visualization

```python
class PortfolioVisualizer:
    """Visualize portfolio allocations and metrics"""
    
    def plot_allocation(self, weights, stock_names):
        """Pie chart of allocation"""
        plt.figure(figsize=(10, 8))
        plt.pie(weights, labels=stock_names, autopct='%1.1f%%', 
                startangle=90)
        plt.title('Portfolio Allocation')
        plt.axis('equal')
        plt.show()
    
    def plot_sector_allocation(self, weights, sector_mapping):
        """Sector-wise allocation"""
        sector_weights = {}
        for stock, weight in zip(sector_mapping.keys(), weights):
            sector = sector_mapping[stock]
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        plt.figure(figsize=(10, 6))
        plt.bar(sector_weights.keys(), sector_weights.values())
        plt.title('Sector Allocation')
        plt.xlabel('Sector')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, returns):
        """Correlation heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            returns.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1
        )
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        plt.show()
```

---

## 📂 Project Structure

```
portfolio/
├── optimization/
│   ├── mpt_optimizer.py
│   ├── efficient_frontier.py
│   ├── risk_parity.py
│   ├── black_litterman.py
│   └── constrained.py
├── backtesting/
│   ├── backtester.py
│   ├── performance_metrics.py
│   └── comparison.py
├── visualization/
│   ├── allocation_plots.py
│   ├── frontier_plot.py
│   └── correlation_plots.py
├── risk_metrics/
│   ├── var.py
│   ├── cvar.py
│   └── sharpe.py
└── examples/
    ├── sample_portfolio.py
    └── strategy_comparison.py
```

---

## ✅ Success Criteria

- [ ] MPT optimizer implemented
- [ ] Efficient frontier calculated
- [ ] Sharpe ratio optimization working
- [ ] Constraints implemented (sector, position, etc.)
- [ ] Risk parity model implemented
- [ ] Black-Litterman model tested
- [ ] Backtesting framework functional
- [ ] Performance metrics calculated
- [ ] Visualizations created
- [ ] Multiple strategies compared

---

## 🛠️ Tools & Libraries

- **scipy.optimize**: Optimization algorithms
- **numpy/pandas**: Numerical computation
- **matplotlib/seaborn**: Visualization
- **cvxpy**: Convex optimization (optional)
- **PyPortfolioOpt**: Portfolio optimization library
- **empyrical**: Performance metrics

---

## 💡 Example Usage

```python
# Load data
returns = pd.read_csv('processed/returns.csv', index_col='date', parse_dates=True)

# Optimize
optimizer = PortfolioOptimizer(returns)

# Max Sharpe ratio
result = optimizer.optimize_sharpe()
print("Optimal Weights:", result['weights'])
print("Expected Return:", result['performance'][0])
print("Volatility:", result['performance'][1])
print("Sharpe Ratio:", result['performance'][2])

# Calculate efficient frontier
ef = EfficientFrontier(optimizer)
frontier = ef.calculate_frontier(num_portfolios=50)
ef.plot_frontier(frontier, result)

# Backtest
backtester = PortfolioBacktester(returns, prices)
portfolio_history = backtester.backtest_equal_weight('monthly')
metrics = backtester.calculate_metrics(portfolio_history)
print("Backtest Metrics:", metrics)
```

---

## 📊 Example Output

```
Recommended Portfolio:
- GP: 35% (Telecom, high growth)
- SQURPHARMA: 25% (Pharma, stable)
- BRACBANK: 20% (Bank exposure)
- BATBC: 20% (Tobacco, dividends)

Expected Annual Return: 14.2%
Volatility (Std Dev): 10.5%
Sharpe Ratio: 1.35
Max Drawdown: -18.3%

Sector Allocation:
- Telecom: 35%
- Pharma: 25%
- Bank: 20%
- Tobacco: 20%
```

---

**Next Phase**: Phase 12 — LLM Financial Advisor

**Last Updated**: 2026-08-13
