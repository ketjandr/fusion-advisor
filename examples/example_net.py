"""Run from the repository root:

fusion-advisor --model examples/example_net.py \
    --input-shape 16384,1024 --dtype int64 --out-dir generated/example_net --apply
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(1024, 512)
        self.proj2 = nn.Linear(512, 512)

    def forward(self, x):
        x = x * 2.0
        x = F.relu(x)
        x = x + 1.0

        x = self.proj1(x)

        x = F.gelu(x)
        x = torch.sigmoid(x)
        x = x * 0.5

        x = self.proj2(x)

        x = torch.tanh(x)
        x = x * 3.0
        return F.softmax(x, dim=-1)
