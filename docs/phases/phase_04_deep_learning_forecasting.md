# PHASE 4 — Deep Learning Forecasting

**Duration**: 3 Weeks  
**Started**: Week 7  
**Status**: 📝 Pending  
**Goal**: Implement deep learning models for stock prediction

---

## 🎯 Objectives

1. Build neural network architectures for time-series
2. Train on sequential price data
3. Compare against baseline models
4. Tune hyperparameters
5. Select best deep learning model

---

## 🧠 Models to Implement

### **Model 1: LSTM (Long Short-Term Memory)**

**Architecture**:
```python
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output
```

**Configuration**:
- Hidden dim: 128
- Num layers: 3
- Dropout: 0.2
- Sequence length: 60 days
- Learning rate: 0.001
- Optimizer: Adam

**Pros**:
- Captures long-term dependencies
- Handles vanishing gradients
- Well-established for sequences

**Cons**:
- Sequential processing (slow)
- Many parameters
- Prone to overfitting
- Difficult to interpret

---

### **Model 2: GRU (Gated Recurrent Unit)**

**Architecture**:
```python
class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(StockGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Attention mechanism
        weights = self.attention(gru_out)
        context = torch.sum(gru_out * weights, dim=1)
        output = self.fc(context)
        return output
```

**Configuration**:
- Hidden dim: 128
- Num layers: 3
- Dropout: 0.2
- Bidirectional: True
- Sequence length: 60 days

**Pros**:
- Faster than LSTM
- Fewer parameters
- Similar performance to LSTM
- Less overfitting

**Cons**:
- May miss some long-term patterns
- Sequential processing
- Less tested than LSTM

---

### **Model 3: CNN-LSTM Hybrid**

**Architecture**:
```python
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters, lstm_hidden, num_layers, output_dim):
        super(CNNLSTM, self).__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) → (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
```

**Configuration**:
- CNN filters: 64
- LSTM hidden: 128
- Kernel size: 3
- Pool size: 2

**Pros**:
- CNN extracts local patterns
- LSTM captures temporal dependencies
- Hybrid benefits
- Often outperforms pure LSTM

**Cons**:
- More complex
- More hyperparameters
- Slower training

---

## 📊 Training Pipeline

### **Data Preparation**
```python
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=60, features=None):
        self.data = data
        self.seq_len = sequence_length
        self.features = features or ['close', 'volume', 'SMA_20', 'RSI_14']
        self.scaler = StandardScaler()
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Get sequence
        X = self.data[self.features].iloc[idx:idx+self.seq_len].values
        y = self.data['close'].iloc[idx+self.seq_len]
        
        # Normalize
        X = self.scaler.fit_transform(X)
        y = (y - self.scaler.mean_[0]) / self.scaler.scale_[0]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])
```

### **Training Loop**
```python
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'Early stopping at epoch {epoch}')
                break
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.6f} - '
              f'Val Loss: {val_loss:.6f}')
    
    return model
```

---

## 🔬 Experimental Setup

### **Hyperparameter Grid**
```python
configs = {
    'sequence_length': [30, 60, 90],
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128]
}
```

### **Cross-Validation**
```python
# Walk-forward validation
for train_years, test_year in walk_forward_splits(data):
    train_data = data[data.year.isin(train_years)]
    test_data = data[data.year == test_year]
    # Train and evaluate
```

---

## 📈 Comparison with Baselines

### **Performance Table**

| Model | RMSE | MAE | MAPE (%) | R² | Dir. Acc (%) |
|-------|------|-----|----------|-----|--------------|
| Random Forest (Phase 3) | TBD | TBD | TBD | TBD | TBD |
| XGBoost (Phase 3) | TBD | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD | TBD |
| GRU | TBD | TBD | TBD | TBD | TBD |
| CNN-LSTM | TBD | TBD | TBD | TBD | TBD |

---

## 📂 Project Structure

```
models/
└── deep_learning/
    ├── architectures/
    │   ├── lstm.py
    │   ├── gru.py
    │   └── cnn_lstm.py
    ├── dataset.py
    ├── train.py
    ├── evaluate.py
    ├── hyperparameter_search.py
    ├── best_model.pth
    └── results/
        ├── training_curves.png
        ├── predictions/
        └── metrics.json
```

---

## ✅ Success Criteria

- [ ] LSTM model implemented and trained
- [ ] GRU model implemented and trained
- [ ] CNN-LSTM model implemented and trained
- [ ] Hyperparameter tuning completed
- [ ] Models compared with baselines (Phase 3)
- [ ] Training curves visualized
- [ ] Best deep learning model identified
- [ ] Model checkpoints saved
- [ ] Results documented

---

## 🛠️ Tools & Libraries

- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration (if available)
- **TensorBoard**: Training monitoring
- **Weights & Biases (wandb)**: Experiment tracking (optional)
- **scikit-learn**: Evaluation metrics

---

## 📊 Hardware Requirements

### **Minimum**
- CPU: 4 cores
- RAM: 16 GB
- GPU: Not required (CPU training possible)

### **Recommended**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA with 8GB+ VRAM
- Training time: ~2-4 hours per model

---

## 💡 Tips & Best Practices

1. **Normalize data** before training
2. **Use early stopping** to prevent overfitting
3. **Gradient clipping** for RNN stability
4. **Learning rate scheduling** for better convergence
5. **Batch normalization** for CNN layers
6. **Dropout** for regularization
7. **Walk-forward validation** for time-series
8. **Save best checkpoint** based on validation loss

---

**Next Phase**: Phase 5 — Advanced Time-Series Models

**Last Updated**: 2026-08-13
