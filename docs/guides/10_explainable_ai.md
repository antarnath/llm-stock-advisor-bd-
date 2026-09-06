# PHASE 8 — Explainable AI

**Duration**: 1 Week  
**Started**: Week 17  
**Status**: 📝 Pending  
**Goal**: Make predictions interpretable

---

## 🎯 Objectives

1. Implement SHAP for feature importance
2. Implement LIME for local explanations
3. Create explanation API
4. Build visualization dashboard
5. Generate human-readable explanations

---

## 🔍 Technique 1: SHAP (SHapley Additive exPlanations)

### **Overview**
SHAP uses game theory to explain model predictions by computing Shapley values for each feature.

### **Implementation for Tree Models**
```python
import shap

class ShapExplainer:
    def __init__(self, model, model_type='tree'):
        self.model = model
        self.model_type = model_type
        
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'deep':
            self.explainer = shap.DeepExplainer(model, background_data)
        elif model_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                model.predict, 
                background_data
            )
    
    def explain_global(self, X_sample):
        """Global feature importance"""
        shap_values = self.explainer.shap_values(X_sample)
        
        # Summary plot
        shap.summary_plot(shap_values, X_sample)
        
        # Feature importance (mean absolute SHAP)
        importance = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance, shap_values
    
    def explain_local(self, x_instance, X_sample):
        """Explain single prediction"""
        shap_values = self.explainer.shap_values(x_instance)
        
        # Force plot
        shap.force_plot(
            self.explainer.expected_value, 
            shap_values, 
            x_instance
        )
        
        # Waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=x_instance.iloc[0],
                feature_names=x_instance.columns
            )
        )
        
        return shap_values
    
    def explain_interaction(self, x_instance):
        """Compute SHAP interaction values"""
        shap_interaction = self.explainer.shap_interaction_values(x_instance)
        shap.summary_plot(shap_interaction, x_instance)
        return shap_interaction
```

### **Implementation for Deep Learning**
```python
class DeepShapExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.background = background_data
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain(self, X_test, feature_names=None):
        """Explain deep learning predictions"""
        shap_values = self.explainer.shap_values(X_test)
        
        # Visualize
        if len(shap_values.shape) == 3:  # Sequence data
            self.visualize_temporal_shap(shap_values, feature_names)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names)
        
        return shap_values
    
    def visualize_temporal_shap(self, shap_values, feature_names):
        """Visualize SHAP values over time"""
        # Average SHAP values across time
        avg_shap = np.abs(shap_values).mean(axis=(0, 2))
        
        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, avg_shap)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance Over Time')
        plt.tight_layout()
        plt.show()
```

---

## 🔍 Technique 2: LIME (Local Interpretable Model-agnostic Explanations)

### **Implementation**
```python
import lime
import lime.lime_tabular

class LimeExplainer:
    def __init__(self, model, training_data, feature_names, 
                 class_names=None):
        self.model = model
        self.feature_names = feature_names
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names or ['price'],
            mode='regression',
            discretize_continuous=True,
            random_state=42
        )
    
    def explain_instance(self, instance, num_features=10, 
                        num_samples=5000):
        """Explain single prediction"""
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Show in notebook
        explanation.show_in_notebook()
        
        # Get explanation as dict
        explanation_dict = dict(explanation.as_list())
        
        return explanation, explanation_dict
    
    def plot_explanation(self, explanation):
        """Plot LIME explanation"""
        # Extract feature contributions
        features, weights = zip(*explanation.as_list())
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in weights]
        plt.barh(range(len(features)), weights, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Contribution')
        plt.title('LIME Explanation')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
```

### **LIME for Time Series**
```python
class LimeTimeSeriesExplainer:
    def __init__(self, model, training_sequences):
        self.model = model
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_sequences.reshape(
                len(training_sequences), -1
            ),
            mode='regression'
        )
    
    def explain_prediction(self, sequence, num_features=10):
        """Explain time-series prediction"""
        # Flatten sequence
        flat = sequence.flatten().reshape(1, -1)
        
        # Get explanation
        explanation = self.explainer.explain_instance(
            data_row=flat[0],
            predict_fn=lambda x: self.model.predict(
                x.reshape(1, *sequence.shape)
            ),
            num_features=num_features
        )
        
        # Map back to time steps
        feature_weights = {}
        for feature_desc, weight in explanation.as_list():
            # Parse "t-N feature_name"
            parts = feature_desc.split()
            if len(parts) >= 2 and parts[0].startswith('t-'):
                time_step = int(parts[0][2:])
                feature_weights[time_step] = weight
        
        return feature_weights
```

---

## 📊 Combined Explanation System

```python
class HybridExplainer:
    """Combine SHAP and LIME for robust explanations"""
    
    def __init__(self, model, model_type='tree', training_data=None,
                 feature_names=None):
        self.shap_explainer = ShapExplainer(model, model_type)
        self.lime_explainer = LimeExplainer(
            model, training_data, feature_names
        )
    
    def explain(self, x_instance, method='both'):
        """Generate explanation using specified method"""
        results = {}
        
        if method in ['shap', 'both']:
            shap_values = self.shap_explainer.explain_local(
                x_instance.to_frame().T, 
                x_instance.to_frame().T
            )
            results['shap'] = shap_values
        
        if method in ['lime', 'both']:
            lime_exp, lime_dict = self.lime_explainer.explain_instance(
                x_instance.values
            )
            results['lime'] = lime_dict
        
        if method == 'both':
            # Aggregate explanations
            results['combined'] = self.combine_explanations(
                results['shap'], results['lime']
            )
        
        return results
    
    def combine_explanations(self, shap_vals, lime_dict):
        """Weighted combination of SHAP and LIME"""
        # Normalize both to same scale
        shap_abs = np.abs(shap_vals.flatten())
        lime_abs = np.abs(list(lime_dict.values()))
        
        shap_norm = shap_abs / (shap_abs.sum() + 1e-8)
        lime_norm = lime_abs / (lime_abs.sum() + 1e-8)
        
        # Weighted average (SHAP gets more weight as it's more reliable)
        combined = 0.6 * shap_norm + 0.4 * lime_norm
        return combined
```

---

## 💬 Natural Language Explanations

### **Template-Based Explanation Generator**
```python
class ExplanationGenerator:
    def __init__(self, prediction, actual_features, feature_importance):
        self.prediction = prediction
        self.features = actual_features
        self.importance = feature_importance
    
    def generate_explanation(self, top_k=5):
        """Generate human-readable explanation"""
        # Get top contributing features
        top_features = sorted(
            self.importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:top_k]
        
        # Generate explanation
        explanation_parts = []
        
        # Overall prediction
        pred_direction = "increase" if self.prediction > 0 else "decrease"
        explanation_parts.append(
            f"📈 Prediction: Expected {pred_direction} of "
            f"{abs(self.prediction):.2f}%"
        )
        
        # Top reasons
        explanation_parts.append("\n🔑 Key Reasons:")
        for feature, weight in top_features:
            feature_name = self.format_feature_name(feature)
            direction = "supporting" if weight > 0 else "against"
            strength = "strongly" if abs(weight) > 0.5 else "moderately"
            
            explanation_parts.append(
                f"  • {feature_name} ({direction}, {strength}): "
                f"{self.features[feature]:.2f}"
            )
        
        # Confidence
        confidence = self.calculate_confidence(top_features)
        explanation_parts.append(
            f"\n🎯 Confidence: {confidence:.0%}"
        )
        
        return '\n'.join(explanation_parts)
    
    def format_feature_name(self, feature):
        """Format feature names for readability"""
        name_map = {
            'rsi_14': 'RSI (momentum)',
            'macd': 'MACD (trend)',
            'sentiment_positive': 'Positive news sentiment',
            'sentiment_negative': 'Negative news sentiment',
            'volume_ratio': 'Trading volume',
            'pe_ratio': 'P/E ratio',
            'eps_growth': 'EPS growth',
            'volatility_30': '30-day volatility'
        }
        return name_map.get(feature, feature)
    
    def calculate_confidence(self, top_features):
        """Calculate prediction confidence"""
        total_weight = sum(abs(w) for _, w in top_features)
        # Higher concentration in top features = higher confidence
        confidence = min(total_weight / 2.0, 1.0)
        return confidence
```

### **LLM-Enhanced Explanation**
```python
import openai

class LLMExplanationGenerator:
    def __init__(self, model='gpt-4'):
        self.model = model
    
    def generate_explanation(self, prediction, features, importance):
        """Use LLM to generate natural explanation"""
        prompt = f"""Generate a professional, concise financial analysis explanation.

Prediction: {prediction}
Key Features: {features}
Feature Importance: {importance}

Provide an explanation that:
1. States the prediction clearly
2. Explains the top 3 reasons
3. Mentions any risks or caveats
4. Uses appropriate financial terminology
5. Is suitable for an investor audience

Explanation:"""
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message['content']
```

---

## 🎨 Visualization Dashboard

### **Explanation Visualization Component**
```python
class ExplanationVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
    
    def plot_feature_importance(self, importance_dict):
        """Bar chart of feature importance"""
        features, values = zip(*sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15])
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if v > 0 else 'red' for v in values]
        plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), [
            self.format_name(f) for f in features
        ])
        plt.xlabel('Importance Score (SHAP)')
        plt.title('Top 15 Feature Importance')
        plt.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waterfall(self, base_value, shap_values, feature_names):
        """SHAP waterfall plot"""
        from shap.plots import waterfall
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=feature_names,
            feature_names=feature_names
        )
        
        shap.plots.waterfall(explanation)
    
    def plot_decision_path(self, prediction_path):
        """Visualize decision flow"""
        # Plot decision tree-like visualization
        fig = go.Figure(go.Sankey(
            node=dict(label=prediction_path['nodes']),
            link=dict(
                source=prediction_path['sources'],
                target=prediction_path['targets'],
                value=prediction_path['values']
            )
        ))
        return fig
```

---

## 🔌 Explanation API

### **FastAPI Endpoints**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ExplanationRequest(BaseModel):
    stock_code: str
    model_type: str
    features: dict
    method: str = 'both'  # 'shap', 'lime', or 'both'

@app.post("/api/explain/prediction")
async def explain_prediction(request: ExplanationRequest):
    """Generate explanation for a prediction"""
    try:
        # Load model and data
        model = load_model(request.stock_code, request.model_type)
        instance = prepare_instance(request.features)
        
        # Generate explanation
        explainer = HybridExplainer(model, request.model_type)
        explanation = explainer.explain(instance, method=request.method)
        
        # Generate natural language
        nl_explanation = generate_nl_explanation(
            explanation, request.features
        )
        
        return {
            'stock_code': request.stock_code,
            'method': request.method,
            'feature_importance': explanation,
            'natural_language': nl_explanation,
            'visualizations': generate_visualizations(explanation)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explain/global/{stock_code}")
async def global_explanation(stock_code: str, model_type: str):
    """Get global feature importance for a stock"""
    model = load_model(stock_code, model_type)
    data = load_stock_data(stock_code)
    
    explainer = ShapExplainer(model, model_type)
    importance, shap_values = explainer.explain_global(data)
    
    return {
        'stock_code': stock_code,
        'importance': importance.to_dict(),
        'summary_plot': generate_summary_plot(shap_values, data)
    }
```

---

## 📋 Example Output

### **Sample Explanation**
```
📈 Prediction: GP expected to increase 7.2% in next 5 days

🔑 Key Reasons:
  • Positive earnings report (supporting, strongly): 0.85
  • RSI_14 oversold condition (supporting, moderately): 0.45
  • High trading volume (supporting, moderately): 0.38
  • Positive market sentiment (supporting, moderately): 0.35
  • MACD bullish crossover (supporting, slightly): 0.22

⚠️ Risks:
  • High volatility (30-day): 28%
  • Beta vs DSEX: 1.15 (slightly volatile)

🎯 Confidence: 78%
```

---

## 📂 Project Structure

```
explainability/
├── techniques/
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   └── hybrid_explainer.py
├── generation/
│   ├── template_generator.py
│   └── llm_generator.py
├── visualization/
│   ├── feature_importance.py
│   ├── waterfall_plots.py
│   └── decision_paths.py
├── api/
│   └── explanation_api.py
└── examples/
    ├── sample_explanations/
    └── test_cases.py
```

---

## ✅ Success Criteria

- [ ] SHAP implementation working for all model types
- [ ] LIME implementation working for tabular data
- [ ] Hybrid explanation system functional
- [ ] Natural language generation working
- [ ] Explanation API endpoints operational
- [ ] Visualization dashboard components built
- [ ] Sample explanations generated and validated
- [ ] Documentation for using explainers

---

## 🛠️ Tools & Libraries

- **SHAP**: `pip install shap`
- **LIME**: `pip install lime`
- **Transformers-Interpret**: For transformer explanations
- **Captum**: PyTorch-specific attribution
- **OpenAI API**: For LLM explanations
- **Matplotlib/Plotly**: Visualization

---

## 💡 Best Practices

1. **Use SHAP for global** explanations
2. **Use LIME for local** explanations
3. **Combine multiple methods** for robustness
4. **Validate explanations** with domain experts
5. **Make explanations accessible** to non-technical users
6. **Document limitations** of each technique

---

**Next Phase**: Phase 9 — RAG System

**Last Updated**: 2026-08-13
