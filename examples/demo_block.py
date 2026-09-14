"""A transformer-ish block written to exercise every path in the advisor.

    fusion-advisor --model examples/demo_block.py --input-shape 8,512,1024 --yes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.scale = nn.Parameter(torch.ones(d))  # [D] against [B, S, D]

    def forward(self, x):
        h = self.norm(x)
        h = self.fc(h)

        # pointwise chain: the case fusion is actually for
        h = F.gelu(h)
        h = h * 0.5
        h = h + 1.0

        h = self.proj(h)

        # broadcast operand, then a second chain
        h = h * self.scale
        h = F.relu(h)

        # reduction at the boundary
        return F.softmax(h, dim=-1)
