# PHASE 5 — Advanced Time-Series Models

**Duration**: 3 Weeks  
**Started**: Week 10  
**Status**: 📝 Pending  
**Goal**: Implement state-of-the-art transformer-based models

---

## 🎯 Objectives

1. Implement 5+ transformer architectures
2. Establish new benchmarks for DSE
3. Compare with LSTM and baselines
4. Write research contribution #1
5. Publish benchmark paper

---

## 📚 Models to Implement

### **Model 1: Vanilla Transformer**

**Architecture**:
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, 
                 seq_len=60, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # Generate causal mask
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        x = self.transformer_encoder(x, mask=mask.to(x.device))
        # Use last token for prediction
        output = self.decoder(x[:, -1, :])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**Configuration**:
- d_model: 128
- nhead: 8
- num_layers: 4
- dropout: 0.1
- dim_feedforward: 512

---

### **Model 2: Informer**

**Architecture** (ProbSparse Self-Attention):
```python
class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, output_dim):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Informer encoder layers with ProbSparse attention
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return self.decoder(x[:, -1, :])

class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention mechanism from Informer paper"""
    def __init__(self, d_model, nhead, factor=5):
        super().__init__()
        self.factor = factor
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, D = x.shape
        H = self.nhead
        
        Q = self.query_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        
        # ProbSparse: only compute top-u queries
        u = self.factor * int(np.ceil(np.log(L)))
        
        # ... (ProbSparse implementation)
        return output
```

**Configuration**:
- ProbSparse factor: 5
- num_layers: 3
- d_model: 128
- nhead: 8

**Pros**:
- O(L log L) complexity (vs O(L²))
- Handles long sequences (1000+ steps)
- Reduced memory footprint

---

### **Model 3: Autoformer**

**Architecture** (Decomposition + Auto-Correlation):
```python
class AutoformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, output_dim):
        super().__init__()
        
        self.decomp = SeriesDecomposition(kernel_size=25)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Auto-correlation attention blocks
        self.encoder = nn.ModuleList([
            AutoCorrelationLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # Series decomposition
        seasonal, trend = self.decomp(x)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        for layer in self.encoder:
            x = layer(x)
        
        return self.decoder(x[:, -1, :])

class SeriesDecomposition(nn.Module):
    """Series decomposition block from Autoformer"""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Transpose for pooling
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelationAttention(nn.Module):
    """Auto-correlation mechanism using FFT"""
    def forward(self, x):
        # Compute auto-correlation via FFT
        Q_fft = torch.fft.rfft(x, dim=1)
        K_fft = torch.fft.rfft(x, dim=1)
        V_fft = torch.fft.rfft(x, dim=1)
        
        # Auto-correlation
        correlations = Q_fft * K_fft.conj()
        attn = torch.fft.irfft(correlations, dim=1)
        
        # Find top-k delays
        top_k = torch.topk(attn, k=5, dim=1)
        
        return apply_top_k_delays(x, top_k, V_fft)
```

**Pros**:
- Decomposes trend and seasonal
- O(L log L) via FFT
- Better for periodic patterns

---

### **Model 4: PatchTST**

**Architecture** (Patch-based):
```python
class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len, stride, d_model, nhead, 
                 num_layers, output_dim):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Channel-independent processing
        self.head = nn.Linear(d_model, output_dim)
    
    def create_patches(self, x):
        # x: (batch, seq_len, channels)
        B, L, C = x.shape
        
        # Pad if necessary
        if L < self.patch_len:
            x = F.pad(x, (0, 0, 0, self.patch_len - L))
            L = self.patch_len
        
        # Create patches
        num_patches = (L - self.patch_len) // self.stride + 1
        patches = x.unfold(1, self.patch_len, self.stride)
        # patches: (batch, num_patches, channels, patch_len)
        
        return patches
    
    def forward(self, x):
        patches = self.create_patches(x)
        B, P, C, L_p = patches.shape
        
        # Process each channel independently
        # Reshape: (B*P, C, L_p) → (B*P*C, L_p)
        patches = patches.reshape(B*P*C, 1, L_p)
        
        # Embed patches
        x = self.patch_embed(patches)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Final prediction (last patch)
        output = self.head(x[:, -1, :])
        
        return output.reshape(B, -1)
```

**Configuration**:
- patch_len: 16
- stride: 8
- d_model: 128
- nhead: 4
- num_layers: 3

**Pros**:
- Captures local semantic information
- Channel-independent (avoid overfitting)
- State-of-the-art on benchmarks

---

### **Model 5: TimeGPT-inspired**

**Architecture** (Foundation Model Approach):
```python
class TimeGPTInspired(nn.Module):
    """Inspired by TimeGPT - Large-scale time-series foundation model"""
    def __init__(self, input_dim, d_model=512, nhead=16, num_layers=12, 
                 output_dim=1, max_seq_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Large transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Multi-output head (for various horizons)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ) for _ in range(7)  # 7 prediction heads
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use last position
        last = x[:, -1, :]
        
        # Multi-horizon predictions
        outputs = [head(last) for head in self.heads]
        return outputs
```

**Configuration**:
- d_model: 512
- num_layers: 12
- nhead: 16
- Pre-training approach (optional)

---

## 🔬 Experimental Framework

### **Unified Training Pipeline**
```python
class TransformerExperiment:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.model = self.build_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        if self.model_name == 'transformer':
            return TransformerModel(**self.config)
        elif self.model_name == 'informer':
            return InformerModel(**self.config)
        # ... etc
    
    def train(self, train_loader, val_loader, epochs=100):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.config['lr'],
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        # ... training loop
        pass
```

---

## 📊 Benchmark Results Template

### **Comprehensive Comparison**

| Model | RMSE | MAE | MAPE | R² | Train Time | Inference Time |
|-------|------|-----|------|-----|------------|----------------|
| Linear Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD | TBD | TBD |
| GRU | TBD | TBD | TBD | TBD | TBD | TBD |
| CNN-LSTM | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD | TBD | TBD |
| Informer | TBD | TBD | TBD | TBD | TBD | TBD |
| Autoformer | TBD | TBD | TBD | TBD | TBD | TBD |
| PatchTST | TBD | TBD | TBD | TBD | TBD | TBD |
| TimeGPT-inspired | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 📂 Project Structure

```
models/
└── transformer/
    ├── architectures/
    │   ├── transformer.py
    │   ├── informer.py
    │   ├── autoformer.py
    │   ├── patchtst.py
    │   └── timegpt_inspired.py
    ├── utils/
    │   ├── positional_encoding.py
    │   ├── attention.py
    │   └── decomposition.py
    ├── train.py
    ├── evaluate.py
    ├── benchmark.py
    └── results/
        ├── best_models/
        ├── predictions/
        └── visualizations/
```

---

## 🎯 Research Contribution #1

**Paper Title**: "Comprehensive Benchmark of Deep Time-Series Models on DSE"

**Abstract Structure**:
- **Background**: Time-series forecasting importance
- **Research Gap**: No comprehensive benchmark for DSE
- **Methodology**: 9+ models on 30+ stocks
- **Results**: Comprehensive comparison table
- **Contribution**: First systematic benchmark
- **Impact**: Reference for future research

**Target Venues**:
- NeurIPS / ICML Workshops
- AAAI Conference
- KDD Workshop on Finance
- Journal of Financial Economics

---

## ✅ Success Criteria

- [ ] All 5 transformer models implemented
- [ ] All models trained on 30+ stocks
- [ ] Comprehensive benchmark table created
- [ ] Statistical tests performed
- [ ] Best model identified
- [ ] Paper draft written
- [ ] Results documented
- [ ] GitHub repository ready for paper

---

## 💡 Implementation Tips

1. **Use HuggingFace** for pre-built components
2. **Mixed precision training** (fp16) for speed
3. **Gradient accumulation** for large models
4. **EMA weights** for better generalization
5. **Learning rate warmup** for transformers
6. **Proper masking** for causal attention

---

## 🛠️ Tools & Libraries

- **PyTorch**: Core framework
- **HuggingFace Transformers**: Pre-built components
- **einops**: Tensor operations
- **timm**: Model implementations
- **neuralforecast**: Time-series models
- **tsai**: Time-series library

---

**Next Phase**: Phase 6 — Sentiment Analysis

**Last Updated**: 2026-08-13
