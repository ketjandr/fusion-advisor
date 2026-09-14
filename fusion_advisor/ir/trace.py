from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from torch.fx.proxy import TraceError

# Traced through, not treated as leaves. Must be fusion-relevant AND
# control-flow-free (nn.MultiheadAttention is neither, it currently raises TraceError).
TRANSPARENT_MODULES: tuple[type, ...] = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.Sigmoid, nn.Tanh, nn.Hardswish,
    nn.Dropout,
    nn.LayerNorm, nn.Softmax,
    nn.Linear,
)


class TracingError(Exception):
    """Model could not be statically traced."""


class FusionTracer(torch.fx.Tracer):
    """fx.Tracer that traces through TRANSPARENT_MODULES and records stack traces."""

    def __init__(self, transparent: tuple[type, ...] = TRANSPARENT_MODULES):
        super().__init__()
        self.transparent = transparent
        self.record_stack_traces = True  # stack tracing for line-number provenance

    @override
    def is_leaf_module(self, m: nn.Module, qualname: str) -> bool:
        """True stops at the module (opaque call_module), False traces inside it."""
        if isinstance(m, self.transparent):
            return False
        return super().is_leaf_module(m, qualname)


def trace(model: nn.Module) -> torch.fx.GraphModule:
    """Trace `model` into a GraphModule, or raise TracingError."""
    tracer = FusionTracer()
    try:
        fx_graph = tracer.trace(model)
    except TraceError as e:
        raise TracingError(
            f"Cannot trace {type(model).__name__}: data-dependent control flow "
            f"in forward(). {e}"
        ) from e
    except Exception as e:  # forward() runs arbitrary user code
        raise TracingError(
            f"Cannot trace {type(model).__name__}: {type(e).__name__}: {e}. "
            f"Shapes are symbolic while tracing, so indexing or reshaping with a "
            f"value read from .shape fails here."
        ) from e

    return torch.fx.GraphModule(model, fx_graph)
