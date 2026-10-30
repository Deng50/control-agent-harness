"""Extracted numerical code. See reuse-manifest.json; original registries are excluded."""
from __future__ import annotations
import math
import random
from typing import Sequence
import torch
from torch import nn

class RNNNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int=128, num_layers: int=2, dropout: float=0.1, static_features_dim: int=5, preview_len: int=50, preview_features_dim: int=1):
        super().__init__()
        expected = static_features_dim + preview_len * preview_features_dim
        if expected != input_dim:
            raise ValueError(f'RNN 配置不一致: {expected} != input_dim({input_dim})')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.static_dim = static_features_dim
        self.preview_len = preview_len
        self.preview_features_dim = preview_features_dim
        per_step_dim = static_features_dim + preview_features_dim
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(input_size=per_step_dim, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='tanh', dropout=rnn_dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, output_dim))

    @classmethod
    def build(cls, input_dim: int, output_dim: int, **hparams) -> 'RNNNetwork':
        return cls(input_dim=input_dim, output_dim=output_dim, **hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'RNN 输入需为 [B, {self.input_dim}]，得到 {tuple(x.shape)}')
        static = x[:, :self.static_dim]
        preview = x[:, self.static_dim:].reshape(x.shape[0], self.preview_len, self.preview_features_dim)
        static_rep = static.unsqueeze(1).expand(-1, self.preview_len, -1)
        seq = torch.cat([static_rep, preview], dim=-1)
        (out, _) = self.rnn(seq)
        return self.head(out[:, -1, :])
