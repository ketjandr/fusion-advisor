"""nn.Module fixtures with hand-known expected clusterings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElementwiseChain(nn.Module):
    """Happy path: one clean fusable chain."""

    def forward(self, x):
        x = x * 2.0
        x = F.relu(x)
        x = x + 1.0
        return x


class ReconvergingDiamond(nn.Module):
    """Forks and rejoins - still one cluster, h stays in a register."""

    def forward(self, x):
        h = F.relu(x)
        return h * 2.0 + h  # catches a naive len(users) > 1 fan-out check


class RepeatedOperand(nn.Module):
    """One user, two arg positions."""

    def forward(self, x):
        h = F.relu(x)
        return h + h


class TrueFanOut(nn.Module):
    """matmul must read h from memory, so h can't be fused away."""

    def __init__(self, d=64):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d))

    def forward(self, x):
        h = F.relu(x)
        return h * 2.0 + h @ self.w


class EscapingOutput(nn.Module):
    """h is returned, so it must exist in memory."""

    def forward(self, x):
        h = F.relu(x)
        return h * 2.0, h


class BroadcastBias(nn.Module):
    """[D] bias against [B, S, D] - forces per-operand index derivation."""

    def __init__(self, d=64):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return F.gelu(x + self.bias)


class ReductionBoundary(nn.Module):
    """scale → mask → softmax."""

    def forward(self, x, mask):
        x = x * 0.125
        x = x.masked_fill(mask, -1e9)
        return F.softmax(x, dim=-1)


class NegativeInfinityMask(nn.Module):
    """Causal-attention spelling: masked logits use negative infinity."""

    def forward(self, x, mask):
        x = x * 0.125
        x = x.masked_fill(mask, float("-inf"))
        return F.softmax(x, dim=-1)


class SumReduction(nn.Module):
    """Reduction that collapses the row, so the store is one scalar per program."""

    def forward(self, x):
        return (x * 2.0).sum(-1)


class ResidualReadTwice(nn.Module):
    """A transformer residual: `h` feeds both the next norm and its own add."""

    def __init__(self, d=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = x + 1.0  # stands in for the attention residual
        return h + F.gelu(self.norm(h))


class UnlowerableOp(nn.Module):
    """Model containing a pointwise op with no supported lowering."""

    def forward(self, x):
        h = F.relu(x)
        h = h.float()
        return h * 2.0


class GeluDropout(nn.Module):
    """One real op plus eval dropout - nothing to fuse."""

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(F.gelu(x))


class RMSNorm(nn.Module):
    """Hand-written RMS norm - one mid-chain reduction, broadcast back over the row."""

    def __init__(self, d=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d))

    def forward(self, x):
        h = x.pow(2).mean(-1, keepdim=True)
        h = x * torch.rsqrt(h + 1e-6)
        return h * self.weight


class FractionalPower(nn.Module):
    """Non-integer exponent - needs a real pow, not repeated multiplies."""

    def forward(self, x):
        h = x.abs() ** 1.5
        return h * 2.0


class TensorPower(nn.Module):
    """Exponent is itself a tensor."""

    def forward(self, x, y):
        h = x.abs() ** y
        return h * 2.0


class PassedDownBuffer(nn.Module):
    """Buffer read in the outer forward, used under a local name in the inner one."""

    class Inner(nn.Module):
        def forward(self, x, mask):
            h = x * 2.0
            return h.masked_fill(mask, 0.0)

    def __init__(self, d=64):
        super().__init__()
        self.inner = self.Inner()
        self.register_buffer("mask", torch.zeros(d, dtype=torch.bool))

    def forward(self, x):
        return self.inner(x, self.mask)


class OpaqueBarrier(nn.Module):
    """matmul between two chains - must yield TWO clusters."""

    def __init__(self, d=64):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d))

    def forward(self, x):
        x = F.relu(x * 2.0)
        x = x @ self.w
        return F.gelu(x + 1.0)


class Untraceable(nn.Module):
    """Data-dependent control flow."""

    def forward(self, x):
        if x.sum() > 0:
            return F.relu(x)
        return x


class ShapeBug(nn.Module):
    """Traces fine (Proxies don't check shapes), fails at ShapeProp."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(7, 9))

    def forward(self, x):
        return F.relu(x @ self.w)


class MutationAfterRead(nn.Module):
    """`h` is read by mul, THEN mutated - FX models dataflow, not mutation.

    No graph edge orders mul before add_, so reordering across it silently
    changes the result. OPAQUE stops add_ joining a cluster but not that.
    """

    def forward(self, x):
        h = F.relu(x)
        y = h * 2.0
        h.add_(1.0)
        return y + h


class ViewAlias(nn.Module):
    """`v` shares storage with `h`."""

    def forward(self, x):
        h = F.relu(x)
        v = h.view(-1)
        return v * 2.0


class ModuleStyle(nn.Module):
    """Every op behind an nn.Module - 4 opaque nodes under symbolic_trace, 0 under FusionTracer."""

    def __init__(self, d=32):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc(h)
        h = self.act(h)
        h = self.drop(h)
        return h + x


class HasUnlistedModule(nn.Module):
    """nn.MultiheadAttention isn't on the allowlist and must stay a leaf."""

    def __init__(self, d=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.act = nn.GELU()

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        return self.act(x + a)


class SharedInput(nn.Module):
    """x feeds both members - one load, not two."""

    def forward(self, x):
        return x * 2.0 + x


class ChainedOneLiner(nn.Module):
    """Every op on one line - the line also holds a matmul, so no safe diff."""

    def __init__(self, d=32):
        super().__init__()
        self.w1, self.act = nn.Linear(d, d), nn.GELU()

    def forward(self, x):
        return self.act(self.w1(x)) + x


class NestedBlock(nn.Module):
    """Ops authored in the inner forward, residual in the outer one."""

    class FFN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.w = nn.Linear(d, d)

        def forward(self, x):
            h = self.w(x)
            h = F.gelu(h)
            return h * 0.5

    def __init__(self, d=32):
        super().__init__()
        self.ffn = self.FFN(d)

    def forward(self, x):
        return self.ffn(x) + x  # residual written at THIS level


class CustomSubmodule(nn.Module):
    """User-defined submodule - fx traces into these by default."""

    class Inner(nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    def __init__(self):
        super().__init__()
        self.inner = self.Inner()

    def forward(self, x):
        return self.inner(x)


class _ScaleBlock(nn.Module):
    """Linear, then a scale/relu/+1 chain - one cluster per instance."""

    def __init__(self, d_in=64, d_out=64, scale=2.0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.scale = scale

    def forward(self, x):
        h = self.fc(x)
        h = h * self.scale
        h = F.relu(h)
        return h + 1.0


class RepeatedBlocks(nn.Module):
    """Two identical blocks - two clusters on the same lines, one kernel."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ScaleBlock(), _ScaleBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PerLayerScale(nn.Module):
    """Same lines, different baked-in constant - must not share a kernel."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ScaleBlock(scale=2.0), _ScaleBlock(scale=3.0)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VaryingWidthBlocks(nn.Module):
    """Same lines, different shapes - must not share a kernel."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ScaleBlock(64, 32), _ScaleBlock(32, 16)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
