"""Merge clusters that repeat per module instance into one kernel and one diff."""

from __future__ import annotations

import torch.fx as fx

from ..sourcemap.provenance import deepest_user_frame
from .cluster import FusableCluster


def signature(cluster: FusableCluster, specs) -> tuple | None:
    """Hashable key equal across instances of the same cluster, or None if unmergeable."""
    position = {n: i for i, n in enumerate(cluster.nodes)}
    input_index = {n: k for k, n in enumerate(cluster.inputs)}

    def operand(a):
        if isinstance(a, fx.Node):
            return ("node", position[a]) if a in position else ("in", input_index[a])
        return ("const", repr(a))  # per-layer constants must not share a kernel

    key = []
    for n in cluster.nodes:
        frame = deepest_user_frame(getattr(n, "stack_trace", None))
        spec = specs.get(n.name)
        if frame is None or spec is None:
            return None
        key.append((
            n.op, str(n.target), frame.path, frame.line,
            tuple(operand(a) for a in n.args),
            tuple(sorted((k, operand(v)) for k, v in n.kwargs.items())),
            spec.dims, spec.dtype,
        ))
    for n in cluster.inputs:
        spec = specs.get(n.name)
        if spec is None:
            return None
        key.append((spec.dims, spec.dtype))
    return tuple(key)  # never owner_path - that is what differs


def merge_twins(clusters: list[FusableCluster], specs) -> list[FusableCluster]:
    """Keep the first cluster per signature, attach the rest as its instances, reindex."""
    kept: list[FusableCluster] = []
    first: dict[tuple, FusableCluster] = {}
    for c in clusters:
        sig = signature(c, specs)
        if sig is not None and sig in first:
            first[sig].instances.append(c)
            continue
        if sig is not None:
            first[sig] = c
        kept.append(c)

    for i, c in enumerate(kept):
        c.index = i
    return kept
