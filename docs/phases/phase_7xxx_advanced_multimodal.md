# Phase 7xxx — Advanced Multimodal Fusion (Attention, Macro, Per-Event)

## Status: 📝 DEFERRED

This phase was originally bundled with Phase 7 (Multimodal Forecasting) but
deferred to keep Phase 7 simple and reproducible on CPU.

Phase 7 (current) implements two fusion strategies:
- **Early fusion** (`MultimodalLSTMEarly`): concat price + sentiment at every
  timestep, single LSTM, 356K params.
- **Late fusion** (`MultimodalLSTMLate`): separate price LSTM (128 hidden, 2
  layers) + small sentiment LSTM (32 hidden, 1 layer), concat final hidden
  states through MLP, 228K params.

Both use the same 7 numeric daily sentiment features
(`n_articles, mean_score, weighted_score, mean_confidence, pos_count,
neg_count, neu_count`) from Phase 6.

**Outcome**: 30 stocks × 2 fusions trained (60 models). Results in
`results/multimodal/multimodal_results.csv`, with apples-to-apples comparison
vs Phase 4 LSTM (`results/multimodal/plots/03_ablation_phase4_vs_phase7.png`)
and per-stock sentiment contribution
(`results/multimodal/plots/06_sentiment_contribution.png`).

---

## Deferred Work

### 1. Attention-based fusion (`MultimodalAttentionFusion`)

Use a cross-attention layer where sentiment features attend over price
features (or vice versa). Implementation:
```python
class MultimodalAttentionFusion(MultimodalBase):
    def __init__(self, price_dim, sent_dim, hidden=128, n_heads=4):
        self.price_proj = nn.Linear(price_dim, hidden)
        self.sent_proj = nn.Linear(sent_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads,
                                                 batch_first=True)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, price, sentiment):
        p = self.price_proj(price)      # (B, seq, H)
        s = self.sent_proj(sentiment)   # (B, seq, H)
        # sentiment attends to price
        attn_out, _ = self.cross_attn(s, p, p)
        out, _ = self.lstm(attn_out)
        return self.head(out[:, -1, :])
```

**Expected impact**: modest — Phase 6 sentiment is sticky/forward-filled,
so cross-attention has limited dynamic content to learn from.

### 2. Macro / DSEX sentiment as market regime signal

Add DSEX-level aggregated sentiment (60-day window of DSEX's `weighted_score`)
as a third input stream that conditions both the price and sentiment branches.
Captures macro regime shifts that affect all stocks.

```python
# In prepare_multimodal_sequences:
dsex_sentiment = (
    sentiment_df[sentiment_df["stock"] == "DSEX"]
    .set_index("date")[["weighted_score"]]
    .reindex(df["date"])
    .ffill()
    .bfill()
    .fillna(0.0)
)
df["dsex_sentiment"] = dsex_sentiment.values
```

### 3. Per-event / article-level sentiment features

Currently we aggregate to daily means. Per-event features could include:
- `max_positive_event_score` (largest single positive article that day)
- `event_diversity` (number of distinct event_types covered)
- `time_since_last_news` (days since the most recent article)
- `event_severity` (max |score| of any article)

These would require re-aggregating from `results/sentiment/news_scored.csv`
(not just `stock_daily_sentiment.csv`) and would add 4-5 features per day.

### 4. Multimodal GRU + CNN-LSTM

Mirror Phase 4xxx (deferred GRU/CNN-LSTM work) but in the multimodal
context. Add `MultimodalGRUEarly` and `MultimodalGRULate` architectures.

---

## When to Pick Up

Re-open this phase when:
1. A real (non-synthetic) news dataset becomes available, OR
2. The user has GPU access for transformer-scale attention layers, OR
3. The thesis moves to per-stock deep analysis where attention interpretability
   would matter.

Suggested next step: implement `MultimodalAttentionFusion` first (cleanest
research story, smallest implementation cost).

---

## Files Touched (when resumed)

**New**:
- `src/training/architectures/multimodal_attn.py`
- `src/training/architectures/multimodal_macro.py`

**Modified**:
- `src/data_processing/multimodal_dataset.py` (add macro merge + per-event features)
- `src/training/multimodal_trainer.py` (register new archs)
- `src/inference/mm_predict.py` (load by arch)
- `src/evaluation/mm_visualize.py` (attention heatmap plots)