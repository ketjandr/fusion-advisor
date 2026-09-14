"""Run from the repository root:

fusion-advisor --model examples/microgpt.py \
    --input-shape 32,256 --dtype int64 --out-dir generated/microgpt -- apply
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroGPT(nn.Module):
    """A two-layer decoder-only transformer with approximately 0.5M parameters."""

    VOCAB_SIZE = 256
    CONTEXT_LENGTH = 256
    D_MODEL = 128
    N_HEADS = 4
    HEAD_DIM = D_MODEL // N_HEADS
    N_LAYERS = 2
    MLP_WIDTH = 4 * D_MODEL
    DROPOUT = 0.1

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(MicroGPT.D_MODEL, MicroGPT.D_MODEL, bias=False)
            self.k_proj = nn.Linear(MicroGPT.D_MODEL, MicroGPT.D_MODEL, bias=False)
            self.v_proj = nn.Linear(MicroGPT.D_MODEL, MicroGPT.D_MODEL, bias=False)
            self.out_proj = nn.Linear(MicroGPT.D_MODEL, MicroGPT.D_MODEL, bias=False)
            self.attn_dropout = nn.Dropout(MicroGPT.DROPOUT)
            self.scale = 1.0 / math.sqrt(MicroGPT.HEAD_DIM)
            self.register_buffer(
                "causal_mask",
                torch.ones(
                    1,
                    1,
                    MicroGPT.CONTEXT_LENGTH,
                    MicroGPT.CONTEXT_LENGTH,
                    dtype=torch.bool,
                ).triu(1),
            )

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(-1, MicroGPT.CONTEXT_LENGTH, MicroGPT.N_HEADS, MicroGPT.HEAD_DIM)
            k = k.view(-1, MicroGPT.CONTEXT_LENGTH, MicroGPT.N_HEADS, MicroGPT.HEAD_DIM)
            v = v.view(-1, MicroGPT.CONTEXT_LENGTH, MicroGPT.N_HEADS, MicroGPT.HEAD_DIM)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k_t = k.transpose(-2, -1)
            scores = q @ k_t
            scores = scores * self.scale
            scores = scores.masked_fill(self.causal_mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = self.attn_dropout(probs)

            context = probs @ v
            context = context.transpose(1, 2)
            context = context.contiguous()
            context = context.view(-1, MicroGPT.CONTEXT_LENGTH, MicroGPT.D_MODEL)
            return self.out_proj(context)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(MicroGPT.D_MODEL, MicroGPT.MLP_WIDTH)
            self.down_proj = nn.Linear(MicroGPT.MLP_WIDTH, MicroGPT.D_MODEL)
            self.activation_dropout = nn.Dropout(MicroGPT.DROPOUT)

        def forward(self, x):
            x = self.up_proj(x)
            x = F.gelu(x)
            x = self.activation_dropout(x)
            return self.down_proj(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(MicroGPT.D_MODEL)
            self.attn = MicroGPT.CausalSelfAttention()
            self.residual_dropout_1 = nn.Dropout(MicroGPT.DROPOUT)
            self.ln_2 = nn.LayerNorm(MicroGPT.D_MODEL)
            self.mlp = MicroGPT.MLP()
            self.residual_dropout_2 = nn.Dropout(MicroGPT.DROPOUT)

        def forward(self, x):
            residual = x
            x = self.ln_1(x)
            x = self.attn(x)
            x = self.residual_dropout_1(x)
            x = x + residual

            residual = x
            x = self.ln_2(x)
            x = self.mlp(x)
            x = self.residual_dropout_2(x)
            return x + residual

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(self.VOCAB_SIZE, self.D_MODEL)
        self.position_embedding = nn.Embedding(self.CONTEXT_LENGTH, self.D_MODEL)
        self.embedding_dropout = nn.Dropout(self.DROPOUT)
        self.blocks = nn.ModuleList([self.Block() for _ in range(self.N_LAYERS)])
        self.final_norm = nn.LayerNorm(self.D_MODEL)
        self.lm_head = nn.Linear(self.D_MODEL, self.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("positions", torch.arange(self.CONTEXT_LENGTH))

    def forward(self, token_ids):
        tokens = self.token_embedding(token_ids)
        positions = self.position_embedding(self.positions)
        x = tokens + positions
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)
