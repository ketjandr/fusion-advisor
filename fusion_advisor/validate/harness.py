"""Subgraph extraction and in-memory model rewrite, for measurement only.

The rewrite is never written to disk or shown to the user - this tool emits a
kernel and a diff, the human applies the edit. It exists here so we can measure
whole-model speedup and catch substitution bugs a subgraph check can't see.
"""

from __future__ import annotations

import torch


def extract_subgraph(gm: torch.fx.GraphModule, cluster) -> torch.fx.GraphModule:
    """Standalone GraphModule computing exactly this cluster.

    Deriving the eager baseline from the real graph, rather than hand-writing a
    reference per fixture, stops every speedup being measured against a strawman.
    """
    raise NotImplementedError


def rewrite_with_kernel(gm: torch.fx.GraphModule, cluster, wrapper) -> torch.fx.GraphModule:
    """Copy of `gm` with the cluster replaced by a call to `wrapper`.

    Must deep-copy - the original is still the baseline, and mutating it turns
    the A/B into a comparison of the patched model against itself.
    """
    raise NotImplementedError


def allocate_inputs(cluster, specs, device="cuda") -> list[torch.Tensor]:
    """Fresh inputs matching the cluster's external TensorSpecs.

    Use the real dtype, not a float32 default - it changes both bytes moved and
    whether reductions need fp32 accumulation.
    """
    raise NotImplementedError
