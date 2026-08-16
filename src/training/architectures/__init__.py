"""
Neural network architectures for stock price forecasting (Phase 4).

Currently implemented:
    - StockLSTM  (Phase 4)

Deferred to Phase 4xxx (see docs/phases/phase_4xxx_gru_cnnlstm.md):
    - StockGRU        (with attention head)
    - StockCNNLSTM    (Conv1d + LSTM hybrid)

Usage:
    from src.training.architectures.lstm import StockLSTM
"""

from src.training.architectures.lstm import StockLSTM

__all__ = ["StockLSTM"]
