# PHASE 7 — Multimodal Forecasting

**Duration**: 2 Weeks  
**Started**: Week 15  
**Status**: 📝 Pending  
**Goal**: Combine multiple data sources for better predictions

---

## 🎯 Objectives

1. Combine price, news, and fundamental data
2. Implement multiple fusion strategies
3. Compare fusion methods
4. Show improvement over unimodal
5. Write research contribution #2

---

## 🏗️ Architecture Overview

```
    Price Data ────→ Price Encoder ────┐
                                       ├──→ Fusion Layer ──→ Prediction Head
    News Sentiment ─→ News Encoder ────┤
                                       │
    Fundamentals ──→ Fund. Encoder ────┘
```

---

## 🧩 Component 1: Price Encoder

```python
class PriceEncoder(nn.Module):
    """LSTM-based encoder for price sequences"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, price_sequence):
        """
        Args:
            price_sequence: (batch, seq_len, price_features)
        Returns:
            encoded: (batch, hidden_dim)
        """
        lstm_out, (h_n, _) = self.lstm(price_sequence)
        
        # Concatenate final forward and backward hidden states
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        h_concat = torch.cat([h_fwd, h_bwd], dim=1)
        
        # Project and normalize
        encoded = self.layer_norm(self.projection(h_concat))
        return encoded
```

---

## 🧩 Component 2: News Encoder

```python
class NewsEncoder(nn.Module):
    """BERT-based encoder for news sentiment sequences"""
    def __init__(self, pretrained_model='bert-base-uncased', 
                 hidden_dim=128, freeze_bert=False):
        super().__init__()
        
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_hidden = self.bert.config.hidden_size  # 768
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, news_features, news_mask=None):
        """
        Args:
            news_features: (batch, news_seq_len, embedding_dim)
            news_mask: (batch, news_seq_len) attention mask
        Returns:
            encoded: (batch, hidden_dim)
        """
        # Aggregate news (mean pooling with attention)
        if news_mask is not None:
            mask_expanded = news_mask.unsqueeze(-1).float()
            summed = (news_features * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            news_agg = summed / counts
        else:
            news_agg = news_features.mean(dim=1)
        
        # Project
        encoded = self.layer_norm(self.projection(news_agg))
        return encoded
```

---

## 🧩 Component 3: Fundamental Encoder

```python
class FundamentalEncoder(nn.Module):
    """MLP-based encoder for fundamental data"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, fundamentals):
        """
        Args:
            fundamentals: (batch, num_fundamental_features)
        Returns:
            encoded: (batch, hidden_dim)
        """
        return self.encoder(fundamentals)
```

---

## 🔀 Fusion Strategies

### **Strategy 1: Early Fusion (Concatenation)**

```python
class EarlyFusionModel(nn.Module):
    def __init__(self, price_input_dim, fundamental_input_dim, 
                 hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.fundamental_encoder = FundamentalEncoder(
            fundamental_input_dim, hidden_dim
        )
        
        # Fusion: concatenate all encodings
        fusion_dim = hidden_dim * 2  # price + fundamental
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, price_seq, fundamentals):
        price_enc = self.price_encoder(price_seq)
        fund_enc = self.fundamental_encoder(fundamentals)
        
        # Concatenate
        fused = torch.cat([price_enc, fund_enc], dim=1)
        return self.fusion(fused)
```

### **Strategy 2: Late Fusion (Weighted Combination)**

```python
class LateFusionModel(nn.Module):
    def __init__(self, price_input_dim, fundamental_input_dim,
                 hidden_dim=128, output_dim=1):
        super().__init__()
        
        # Separate prediction heads for each modality
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.price_predictor = nn.Linear(hidden_dim, output_dim)
        
        self.fundamental_encoder = FundamentalEncoder(
            fundamental_input_dim, hidden_dim
        )
        self.fundamental_predictor = nn.Linear(hidden_dim, output_dim)
        
        # Learnable weights for fusion
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, price_seq, fundamentals):
        price_enc = self.price_encoder(price_seq)
        fund_enc = self.fundamental_encoder(fundamentals)
        
        price_pred = self.price_predictor(price_enc)
        fund_pred = self.fundamental_predictor(fund_enc)
        
        # Weighted combination
        weights = self.softmax(self.fusion_weights)
        fused_pred = (weights[0] * price_pred + 
                     weights[1] * fund_pred)
        
        return fused_pred
```

### **Strategy 3: Attention-Based Fusion (Recommended)**

```python
class AttentionFusionModel(nn.Module):
    def __init__(self, price_input_dim, fundamental_input_dim,
                 hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.fundamental_encoder = FundamentalEncoder(
            fundamental_input_dim, hidden_dim
        )
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, price_seq, fundamentals):
        # Encode modalities
        price_enc = self.price_encoder(price_seq).unsqueeze(1)  # (B, 1, H)
        fund_enc = self.fundamental_encoder(fundamentals).unsqueeze(1)  # (B, 1, H)
        
        # Stack for attention
        modalities = torch.cat([price_enc, fund_enc], dim=1)  # (B, 2, H)
        
        # Self-attention across modalities
        attended, attn_weights = self.cross_attention(
            modalities, modalities, modalities
        )
        
        # Flatten and concatenate
        price_flat = price_enc.squeeze(1)
        fund_flat = fund_enc.squeeze(1)
        attended_flat = attended.mean(dim=1)
        
        fused = torch.cat([price_flat, fund_flat, attended_flat], dim=1)
        return self.fusion(fused)
```

### **Strategy 4: Tensor Fusion**

```python
class TensorFusionModel(nn.Module):
    """Tensor fusion network for multimodal learning"""
    def __init__(self, price_input_dim, fundamental_input_dim,
                 hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.fundamental_encoder = FundamentalEncoder(
            fundamental_input_dim, hidden_dim
        )
        
        # Tensor fusion: outer product
        fusion_dim = (hidden_dim + 1) * (hidden_dim + 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, price_seq, fundamentals):
        price_enc = self.price_encoder(price_seq)
        fund_enc = self.fundamental_encoder(fundamentals)
        
        # Append 1 for tensor fusion
        price_1 = torch.cat([price_enc, torch.ones_like(price_enc[:, :1])], dim=1)
        fund_1 = torch.cat([fund_enc, torch.ones_like(fund_enc[:, :1])], dim=1)
        
        # Outer product
        batch_size = price_1.size(0)
        fused = torch.bmm(
            price_1.unsqueeze(2), 
            fund_1.unsqueeze(1)
        ).view(batch_size, -1)
        
        return self.fusion(fused)
```

---

## 📊 Full Multimodal Architecture with News

```python
class FullMultimodalModel(nn.Module):
    """Complete multimodal model: Price + News + Fundamentals"""
    def __init__(self, price_input_dim, news_embedding_dim, 
                 fundamental_input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.news_encoder = NewsEncoder(hidden_dim=hidden_dim)
        self.fundamental_encoder = FundamentalEncoder(
            fundamental_input_dim, hidden_dim
        )
        
        # Cross-modal attention
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final prediction head
        fusion_dim = hidden_dim * 4  # 3 modalities + attention context
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, price_seq, news_features, fundamentals, 
                news_mask=None):
        # Encode each modality
        price_enc = self.price_encoder(price_seq).unsqueeze(1)
        news_enc = self.news_encoder(news_features, news_mask).unsqueeze(1)
        fund_enc = self.fundamental_encoder(fundamentals).unsqueeze(1)
        
        # Stack modalities
        modalities = torch.cat([price_enc, news_enc, fund_enc], dim=1)
        
        # Cross-modal attention
        attended, _ = self.modality_attention(
            modalities, modalities, modalities
        )
        
        # Flatten all
        all_features = torch.cat([
            price_enc.squeeze(1),
            news_enc.squeeze(1),
            fund_enc.squeeze(1),
            attended.mean(dim=1)
        ], dim=1)
        
        return self.predictor(all_features)
```

---

## 🎓 Training Pipeline

### **Data Preparation**
```python
class MultimodalDataset(Dataset):
    def __init__(self, price_data, news_data, fundamental_data, 
                 seq_len=60, news_window=7):
        self.price_data = price_data
        self.news_data = news_data
        self.fundamental_data = fundamental_data
        self.seq_len = seq_len
        self.news_window = news_window
    
    def __getitem__(self, idx):
        # Get price sequence
        price_seq = self.price_data[idx:idx+self.seq_len]
        target = self.price_data.iloc[idx+self.seq_len]['close']
        
        # Get news for the window
        date = self.price_data.iloc[idx+self.seq_len]['date']
        news_window_start = date - timedelta(days=self.news_window)
        relevant_news = self.news_data[
            (self.news_data['date'] >= news_window_start) &
            (self.news_data['date'] <= date)
        ]
        
        # Aggregate news features (e.g., average sentiment)
        news_features = self.aggregate_news(relevant_news)
        
        # Get fundamentals
        fundamentals = self.fundamental_data.iloc[idx+self.seq_len]
        
        return {
            'price_seq': torch.FloatTensor(price_seq.values),
            'news_features': torch.FloatTensor(news_features),
            'fundamentals': torch.FloatTensor(fundamentals.values),
            'target': torch.FloatTensor([target])
        }
    
    def aggregate_news(self, news_df):
        """Aggregate news sentiment features"""
        if len(news_df) == 0:
            return np.zeros(5)  # [pos, neg, neu, compound, count]
        
        return np.array([
            news_df['positive_score'].mean(),
            news_df['negative_score'].mean(),
            news_df['neutral_score'].mean(),
            news_df['compound_score'].mean(),
            len(news_df) / 10  # Normalized count
        ])
```

---

## 📂 Project Structure

```
models/
└── multimodal/
    ├── encoders/
    │   ├── price_encoder.py
    │   ├── news_encoder.py
    │   └── fundamental_encoder.py
    ├── fusion/
    │   ├── early_fusion.py
    │   ├── late_fusion.py
    │   ├── attention_fusion.py
    │   └── tensor_fusion.py
    ├── models/
    │   ├── full_multimodal.py
    │   └── baseline_comparison.py
    ├── dataset.py
    ├── train.py
    ├── evaluate.py
    └── results/
        ├── ablation_study/
        ├── attention_weights/
        └── visualizations/
```

---

## 🧪 Ablation Study

### **Comparison Table**

| Configuration | RMSE | MAE | R² | Dir. Acc |
|---------------|------|-----|-----|----------|
| Price Only | TBD | TBD | TBD | TBD |
| Price + Fundamentals | TBD | TBD | TBD | TBD |
| Price + News | TBD | TBD | TBD | TBD |
| Price + News + Fundamentals | TBD | TBD | TBD | TBD |

### **Fusion Strategy Comparison**

| Fusion Method | RMSE | MAE | R² | Train Time |
|---------------|------|-----|-----|------------|
| Early Fusion | TBD | TBD | TBD | TBD |
| Late Fusion | TBD | TBD | TBD | TBD |
| Attention Fusion | TBD | TBD | TBD | TBD |
| Tensor Fusion | TBD | TBD | TBD | TBD |

---

## 🎯 Research Contribution #2

**Paper Title**: "Multimodal Stock Forecasting for Bangladesh Market"

**Abstract Structure**:
- **Background**: Single modality limitations
- **Research Gap**: No multimodal approach for DSE
- **Methodology**: Price + News + Fundamentals fusion
- **Results**: 15-25% improvement over unimodal
- **Contribution**: Novel multimodal framework for emerging markets
- **Impact**: Applicable to other emerging markets

**Key Findings to Report**:
1. Multimodal > unimodal (statistically significant)
2. Attention fusion outperforms other methods
3. News sentiment has highest marginal value
4. Fundamentals provide long-term signal

---

## 📊 Visualization

### **Attention Weight Analysis**
```python
def visualize_attention_weights(model, dataloader):
    """Visualize which modalities the model focuses on"""
    model.eval()
    attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, attn = model(
                batch['price_seq'], 
                batch['news_features'], 
                batch['fundamentals']
            )
            attention_weights.append(attn.cpu().numpy())
    
    # Average attention weights
    avg_weights = np.mean(np.concatenate(attention_weights), axis=0)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_weights, annot=True, 
                xticklabels=['Price', 'News', 'Fundamentals'],
                yticklabels=['Price', 'News', 'Fundamentals'])
    plt.title('Cross-Modal Attention Weights')
    plt.show()
```

### **Feature Importance**
```python
def analyze_feature_importance(model, test_data):
    """Analyze contribution of each modality"""
    results = {}
    
    # Full model
    results['full'] = evaluate(model, test_data)
    
    # Without news
    test_no_news = test_data.copy()
    test_no_news['news_features'] = 0
    results['no_news'] = evaluate(model, test_no_news)
    
    # Without fundamentals
    test_no_fund = test_data.copy()
    test_no_fund['fundamentals'] = 0
    results['no_fundamentals'] = evaluate(model, test_no_fund)
    
    # Price only
    results['price_only'] = evaluate_price_only(model, test_data)
    
    return results
```

---

## ✅ Success Criteria

- [ ] All encoders implemented
- [ ] 4 fusion strategies implemented
- [ ] Multimodal model trained successfully
- [ ] Ablation study completed
- [ ] Attention weights visualized
- [ ] Improvement over unimodal demonstrated (statistical test)
- [ ] Research paper draft written
- [ ] Results documented

---

## 🛠️ Tools & Libraries

- **PyTorch**: Deep learning
- **Transformers**: News encoder
- **SHAP**: Feature importance
- **Matplotlib/Seaborn**: Visualization

---

## 💡 Tips

1. **Normalize each modality** separately
2. **Handle missing news** with zero vectors
3. **Use gradient clipping** for stability
4. **Ensemble multiple fusion methods** for robustness
5. **Validate fusion strategy** on validation set

---

**Next Phase**: Phase 8 — Explainable AI

**Last Updated**: 2026-08-13

---

## ✅ IMPLEMENTATION COMPLETE (Aug 2026)

**Status**: ✅ Phase 7 complete; attention fusion / macro features / per-event features deferred to [phase_7xxx_advanced_multimodal.md](phase_7xxx_advanced_multimodal.md).

### What was actually built (vs the original plan above)

| Original plan | Actual implementation | Why |
|---|---|---|
| News encoder Transformer | Two LSTM fusions: Early + Late | Phase 6 sentiment is forward-filled (sticky), so a transformer would have little dynamic content to attend to. LSTMs suffice. |
| Handle missing news with zero vectors | **Forward-fill per stock (sticky)**, bfill for leading NaNs, 0.0 fallback | Forward-fill is realistic — sentiment decays slowly. Zero-fill would create artificial "no signal" events on 99% of days. |
| Ensemble multiple fusion methods | Two architectures trained separately, results compared in `03_ablation_phase4_vs_phase7.png` | True ensembling deferred — would obscure which fusion wins. |
| Validate fusion strategy | Yes — same time-based 72/8/20 split as Phase 4; apples-to-apples comparison | — |

### Key design decisions

- **Two fusion strategies** for ablation (user-locked):
  - **Early fusion** (`MultimodalLSTMEarly`): concat at every timestep → single LSTM (356K params)
  - **Late fusion** (`MultimodalLSTMLate`): separate LSTMs → concat final hidden → MLP (228K params)
- **Sentiment NaN policy**: per-stock forward-fill, bfill for leading NaNs, 0.0 fallback
- **Inference**: sentiment window FROZEN across all 5 forecast steps (sentiment is exogenous); only price features iterate (Returns_1d/5d/20d/Log_Returns)
- **Two scalers**, both fit on TRAIN only — price scaler and sentiment scaler are separate (avoids polluting the price distribution with sparse sentiment)

### Files created

```
src/
├── data_processing/
│   └── multimodal_dataset.py            # MultimodalSequenceDataset + prepare_multimodal_sequences
├── training/
│   ├── architectures/
│   │   └── multimodal_lstm.py           # MultimodalLSTMEarly + MultimodalLSTMLate + build_multimodal
│   └── multimodal_trainer.py            # trains both fusions, resume support, incremental CSV
├── inference/
│   └── mm_predict.py                    # 5-day predictions, sentiment frozen
└── evaluation/
    └── mm_visualize.py                  # 8 plots + summary report

models/multimodal/
  {STOCK}_best_mm_early.{pt,pkl}         # 30 stocks
  {STOCK}_best_mm_late.{pt,pkl}          # 30 stocks

results/multimodal/
  multimodal_results.csv                 # 30 × 2 = 60 rows
  predictions_5days.csv                  # + fusion_strategy column
  summary_report.txt
  plots/
    01_model_summary.png
    02_per_stock_performance.png
    03_ablation_phase4_vs_phase7.png    # Phase 4 LSTM vs Phase 7 multimodal
    04_confusion_matrix.png
    05_training_curves.png
    06_sentiment_contribution.png       # Δ Dir_Acc per stock
    07_metric_distributions.png
    08_directional_accuracy_bars.png
```

### How to reproduce

```bash
# Train both fusions on all 30 stocks
python src/training/multimodal_trainer.py --fusion both

# Generate 5-day predictions for both fusions
python src/inference/mm_predict.py --fusion both

# 8 plots + summary report + ablation vs Phase 4
python src/evaluation/mm_visualize.py
```

Or in one shot: `python scripts/run_pipeline.py --phase 7`

### Honest expectations

Phase 7 uses **synthetic** sentiment news (Phase 6 was a curated corpus). With only
~50 of ~3000 trading days per stock carrying any news, and the signal being
forward-filled, expect:
- **Modest Δ** vs Phase 4 LSTM (Dir_Acc typically within ±2 percentage points)
- **No leakage** = Dir_Acc stays in 48-52% range (matches Phase 3/4 reality)
- **Late fusion** usually slightly better than early fusion on sparse sentiment
  (fewer parameters, less overfitting)

If Dir_Acc jumps to >55%, that's a leakage bug — debug the sentiment merge
or scaler discipline.
