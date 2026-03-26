"""
模型模块 - 包含GRU模型定义和训练逻辑
"""

from .quant_gru import QuantGRU, ic_loss, train_model

__all__ = [
    'QuantGRU',
    'ic_loss',
    'train_model',
]
