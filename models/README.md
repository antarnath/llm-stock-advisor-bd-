# 🤖 Models

Trained model artifacts organized by phase.

## Structure

```
models/
├── baseline/              # Phase 3: Classical ML models
│   ├── ACI_best.pkl      #   30 trained models (one per DSE stock)
│   └── ...
│
├── deep_learning/         # Phase 4: LSTM, GRU, CNN-LSTM (PyTorch checkpoints)
│
├── transformers/          # Phase 5: Informer, Autoformer, PatchTST
│
└── experiments/           # Phase 7: Multimodal fusion experiments
```

## Naming Conventions

- **Baseline models:** `{STOCK_CODE}_best.pkl` (pickled sklearn/xgboost model)
- **Deep learning:** `{STOCK_CODE}_{model_name}_v{version}.pt` (PyTorch state dict)
- **Transformers:** `{model_name}_{dataset}_{timestamp}.ckpt` (Lightning checkpoints)

## Loading a Model

```python
import pickle
from pathlib import Path

model_path = Path("models/baseline/ACI_best.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]
print(f"Loaded model with {len(features)} features")
```

## Gitignore

All model files (`.pkl`, `.pt`, `.h5`, `.onnx`) are gitignored by default — they're large and reproducible from code. Use DVC or external storage (S3/GCS) for sharing.

## Versioning

For deep learning models, use [DVC](https://dvc.org/) or [Git LFS](https://git-lfs.com/) for proper version control. Each experiment should log:
- Hyperparameters
- Training metrics
- Test metrics
- Git commit hash