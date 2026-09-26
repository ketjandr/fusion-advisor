"""Memory traffic estimator: VRAM upper bound, no cache model.

    unfused = every operand read + every result written, per node
    fused   = cluster-external inputs read + escaping outputs written

On a 24 MB L2 most intermediates never leave cache, so measured speedup
lands below this estimate. Defer to benchmarks for real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.fx as fx

from ..ir.op_registry import is_identity


@dataclass
class TrafficEstimate:
    unfused_bytes: int
    fused_bytes: int

    @property
    def savings_ratio(self) -> float:
        """Fraction of traffic eliminated, 0.0 to 1.0."""
        if self.unfused_bytes == 0:
            return 0.0
        return 1.0 - self.fused_bytes / self.unfused_bytes


def _node_bytes(node: fx.Node, specs: dict) -> int:
    """Output size in bytes, or 0 if no spec (non-tensor node)."""
    spec = specs.get(node.name)
    return spec.nbytes if spec else 0


def estimate(cluster, specs) -> TrafficEstimate:
    """Estimate DRAM traffic with and without fusion.

    Unfused: each node reads all its inputs from memory and writes its output.
    Fused: only external inputs are read, only escaping outputs are written.
    """
    unfused = 0
    for n in cluster.nodes:
        if is_identity(n):  # eval dropout aliases its input, so eager moves nothing
            continue
        unfused += _node_bytes(n, specs)  # write to VRAM
        operands = []
        fx.node.map_arg((n.args, n.kwargs), operands.append)  # keeps repeats: h + h reads twice
        unfused += sum(_node_bytes(a, specs) for a in operands)  # read from VRAM

    fused = sum(_node_bytes(n, specs) for n in cluster.inputs)  # read from VRAM
    fused += sum(_node_bytes(n, specs) for n in cluster.outputs)  # write to VRAM

    return TrafficEstimate(unfused_bytes=unfused, fused_bytes=fused)
