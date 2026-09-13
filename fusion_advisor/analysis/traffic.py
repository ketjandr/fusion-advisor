"""Memory traffic estimator: VRAM upper bound, no cache model.

    unfused = every operand read + every result written, per node
    fused   = cluster-external inputs read + escaping outputs written

On a 24 MB L2 most intermediates never leave cache, so measured speedup
lands below this estimate. Defer to benchmarks for real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.fx as fx

from .legality import escaping_nodes


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


def _input_bytes(node: fx.Node, cluster_set: set, specs: dict) -> int:
    """Bytes read by this node from OUTSIDE the cluster."""
    total = 0
    for a in node.args:
        if isinstance(a, fx.Node) and a not in cluster_set:
            total += _node_bytes(a, specs)
    return total


def estimate(cluster, specs) -> TrafficEstimate:
    """Estimate DRAM traffic with and without fusion.

    Unfused: each node reads all its inputs from memory and writes its output.
    Fused: only external inputs are read, only escaping outputs are written.
    The difference is the intermediate traffic that stays in registers.
    """
    cluster_nodes = cluster.nodes
    cluster_set = set(cluster_nodes)
    escaping = escaping_nodes(cluster_nodes)

    unfused = 0
    for n in cluster_nodes:
        unfused += _node_bytes(n, specs)  # write to VRAM
        for a in n.args:
            if isinstance(a, fx.Node):
                unfused += _node_bytes(a, specs)  # read from VRAM

    fused = sum(_input_bytes(n, cluster_set, specs) for n in cluster_nodes) # read from VRAM
    fused += sum(_node_bytes(n, specs) for n in escaping)  # write to VRAM

    return TrafficEstimate(unfused_bytes=unfused, fused_bytes=fused)
