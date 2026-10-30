"""Extracted numerical code. See reuse-manifest.json; original registries are excluded."""
from __future__ import annotations
import math
import random
from typing import Sequence
import torch
from torch import nn

class PerceptionNet(nn.Module):

    def __init__(self, x_dim: int, u_dim: int, hidden: int=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim + u_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, x_dim))

    def forward(self, x, u_phys):
        return self.net(torch.cat([x, u_phys], dim=-1))
