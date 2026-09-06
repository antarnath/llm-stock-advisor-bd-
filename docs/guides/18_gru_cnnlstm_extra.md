# PHASE 4xxx — Deep Learning Extension (GRU + CNN-LSTM) [DEFERRED]

**Status**: � **Deferred — to be completed after Phase 4 (LSTM) ships**
**Depends on**: Phase 4 (LSTM only) must be complete and benchmarked.
**Goal**: Add the remaining two deep-learning architectures from the original Phase 4 spec, train them on the same leak-free data, and produce a 3-way comparison report.

---

## Why this phase exists

The original Phase 4 specification ([phase_04_deep_learning_forecasting.md](phase_04_deep_learning_forecasting.md)) calls for three architectures:

1. ✅ LSTM (Long Short-Term Memory) — **shipped in Phase 4**
2. 📝 GRU (Gated Recurrent Unit, with attention head)
3. 📝 CNN-LSTM hybrid (Conv1d → BatchNorm → ReLU → MaxPool → LSTM)

Phase 4 shipped LSTM only to keep the deep-learning pipeline (sequence dataset → training loop → checkpointing → inference → viz) provable end-to-end first. This phase extends the same trainer class with the two remaining architectures.

---

## Reuse from Phase 4

All the following Phase 4 artifacts are already in place and reused:

- `src/data_processing/sequence_dataset.py` — `StockSequenceDataset`, `prepare_stock_sequences`, `EXCLUDE_COLS`
- `src/training/deep_learning_trainer.py` — `DeepLearningTrainer` class with `train_lstm`, `train_stock`, `train_all_stocks` methods
- `src/training/architectures/lstm.py` — `StockLSTM` reference architecture
- `src/inference/dl_predict.py` — load + predict pattern (parameterized by arch name)
- `src/evaluation/dl_visualize.py` — visualization scaffolding
- `scripts/run_pipeline.py` — `phase_4_deep_learning` dispatch

The trainer class is intentionally structured so adding a new architecture = adding one `train_<arch>` method, one model class, and one `models_dir.glob("*_best_<arch>.pt")` glob in the pipeline runner. **No architectural rework needed.**

---

## What to add

### 1. New architectures

`src/training/architectures/gru.py` — `StockGRU` (with attention over time steps):

```python
class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3,
                 output_dim=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Sequential(nn.Linear(out_dim, 1), nn.Softmax(dim=1))
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)               # (batch, seq, out_dim)
        weights = self.attention(out)      # (batch, seq, 1)
        context = torch.sum(out * weights, dim=1)
        return self.head(context)
```

`src/training/architectures/cnn_lstm.py` — `StockCNNLSTM` (Conv1d feature extractor + LSTM temporal model):

```python
class StockCNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128,
                 num_layers=2, output_dim=1, dropout=0.2, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x: (batch, seq, n_feat) -> (batch, n_feat, seq)
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # (batch, cnn_filters, seq') -> (batch, seq', cnn_filters)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
```

### 2. Trainer additions

In `src/training/deep_learning_trainer.py`:

- Add `ARCHS = ["lstm", "gru", "cnnlstm"]` constant.
- Replace `train_lstm(...)` with a generic `train_<arch>(...)` that selects the model class via `ARCHITECTURES = {"lstm": StockLSTM, "gru": StockGRU, "cnnlstm": StockCNNLSTM}`.
- In `train_stock`, iterate over all three architectures, train each, evaluate, and pick best by test RMSE.
- Save each arch's checkpoint as `{STOCK}_best_<arch>.pt` + sidecar.
- Update result CSV columns to include per-arch metrics + overall best_arch/best_rmse.

### 3. Inference & viz

- `src/inference/dl_predict.py` — generalize to take `--arch {lstm,gru,cnnlstm}` (or run all three, then output one predictions CSV per arch).
- `src/evaluation/dl_visualize.py` — generalize `MODELS` and `COLORS` lists to 3 entries; add a 3-way comparison panel.

### 4. Pipeline runner

- Extend `phase_4_deep_learning()` to also include GRU and CNN-LSTM steps.

### 5. Phase 4 results CSV → Phase 4xxx results CSV

New artifact layout:

```
results/deep_learning/
├── deep_learning_results.csv              # 30 rows × {stock, name, best_arch, best_rmse, LSTM_*, GRU_*, CNNLSTM_*}
├── predictions_5days_lstm.csv             # day-1..5 per stock (LSTM)
├── predictions_5days_gru.csv              # (GRU)
├── predictions_5days_cnnlstm.csv          # (CNN-LSTM)
├── summary_report.txt                     # 3-way comparison narrative
└── plots/
    ├── 01_model_comparison.png            # RMSE/R²/Dir_Acc across 3 archs
    ├── 02_per_stock_performance.png       # per-stock × 3 archs
    ├── 03_best_arch_distribution.png      # which arch wins for each stock
    ├── 04_metrics_distribution.png        # distributions per arch
    ├── 05_directional_accuracy.png        # per-arch Dir_Acc vs 50%
    └── 06_training_curves.png             # avg train/val loss per arch
```

---

## Hyperparameters (reuse Phase 4 defaults where applicable)

| Param | GRU | CNN-LSTM |
|---|---|---|
| sequence_length | 60 | 60 |
| hidden_dim | 128 | 128 (LSTM), 64 (CNN filters) |
| num_layers | 3 | 2 |
| dropout | 0.2 | 0.2 |
| learning_rate | 1e-3 | 1e-3 |
| batch_size | 64 | 64 |
| epochs | 100 | 100 |
| patience | 20 | 20 |
| extras | attention head, bidirectional=False | kernel=3, MaxPool=2 |

If early experiments show GRU/CNN-LSTM underperforming LSTM, **do not** auto-tune hyperparameters to "win" — record the honest result and discuss in the report. Leakage must not creep back in.

---

## Out of scope (still)

- Transformer architectures (Phase 5)
- Multimodal / news / sentiment (Phase 6+)
- Walk-forward cross-validation (single 80/20 split)
- Hyperparameter sweeps (use spec defaults; tune only if a clear failure mode emerges)

---

## Verification (when this phase ships)

1. Smoke test on 1 stock with 5 epochs:
   ```bash
   python src/training/deep_learning_trainer.py --max-stocks 1 --epochs 5
   ```
2. Full run:
   ```bash
   python scripts/run_pipeline.py --phase 4
   ```
3. Confirm Dir_Acc stays in 48–56% range (same as Phase 3 baselines + Phase 4 LSTM). If any arch gets Dir_Acc > 60%, leakage.
4. Compare three archs side-by-side via `results/deep_learning/summary_report.txt`.

---

**Last Updated**: 2026-08-16
**Created by**: Phase 4 implementation (deferred work tracker)
