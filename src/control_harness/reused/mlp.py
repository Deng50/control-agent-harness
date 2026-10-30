"""Extracted numerical code. See reuse-manifest.json; original registries are excluded."""
from __future__ import annotations
import math
import random
from typing import Sequence
import torch
from torch import nn

class MLPNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]=(256, 128, 64), dropout: float=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def build(cls, input_dim: int, output_dim: int, **hparams) -> 'MLPNetwork':
        return cls(input_dim=input_dim, output_dim=output_dim, **hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'MLP 输入需为 [B, {self.input_dim}]，得到 {tuple(x.shape)}')
        return self.net(x)
