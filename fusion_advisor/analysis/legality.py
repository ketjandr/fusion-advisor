from __future__ import annotations

import torch
import torch.fx as fx

from ..codegen.lowering import has_lowering
from ..ir.op_registry import OpCategory, classify, reduction_axis
from ..ir.shapes import Dim
from .cluster import RejectionReason

_VIEW_METHODS = {
    "view", "reshape", "transpose", "permute", "unsqueeze", "squeeze",
    "expand", "narrow", "unflatten", "flatten", "t",
}
_VIEW_FNS = {
    torch.reshape, torch.transpose, torch.unsqueeze, torch.squeeze, torch.flatten,
}


def escaping_nodes(cluster_nodes) -> list:
    """Nodes in the cluster with at least one consumer outside the cluster.

    These are the values the kernel must actually write to memory.
    """
    inside = set(cluster_nodes)
    return [n for n in cluster_nodes if any(u not in inside for u in n.users)]


def external_inputs(cluster_nodes) -> list:
    """Distinct values the kernel must load, in first-use order.

    Deduplicated: two members reading the same tensor is one load, not two.
    """
    inside = set(cluster_nodes)
    seen, out = set(), []
    for n in cluster_nodes:
        for a in n.args:
            if isinstance(a, fx.Node) and a not in inside and a not in seen:
                seen.add(a)
                out.append(a)
    return out


def check_fan_out(cluster_nodes) -> RejectionReason | None:
    """Reject clusters needing more than one stored output.
  
    TODO: support valid multi-external consumers
    """
    return RejectionReason.FAN_OUT if len(escaping_nodes(cluster_nodes)) > 1 else None

def check_convexity(cluster_nodes) -> RejectionReason | None:
    """Reject if a dependency path leaves the cluster and comes back."""
    inside = set(cluster_nodes)
    frontier = [u for n in cluster_nodes for u in n.users if u not in inside]
    visited = set(frontier)
    while frontier:
        node = frontier.pop()
        for u in node.users:
            if u in inside:
                return RejectionReason.NON_CONVEX
            if u not in visited:
                visited.add(u)
                frontier.append(u)
    return None


def _broadcast(a: tuple[Dim, ...], b: tuple[Dim, ...]) -> tuple[Dim, ...] | None:
    """Broadcast output shape, or None if incompatible."""
    ndim = max(len(a), len(b))
    result = []
    for i in range(1, ndim + 1):
        da = a[-i] if i <= len(a) else Dim(1)
        db = b[-i] if i <= len(b) else Dim(1)
        if da.provably_equal(db):
            result.append(da)
        elif da.value == 1:
            result.append(db)
        elif db.value == 1:
            result.append(da)
        else:
            return None
    result.reverse()
    return tuple(result)


def check_shapes(cluster_nodes, specs) -> RejectionReason | None:
    """Reject unless all members are provably broadcast-compatible."""
    shapes = [
        specs[n.name].shape
        for n in cluster_nodes
        if n.name in specs and classify(n) is not OpCategory.REDUCTION  # reduction is exempt
    ]
    if len(shapes) < 2:
        return None
    out = shapes[0]
    for s in shapes[1:]:
        out = _broadcast(out, s)
        if out is None:
            return RejectionReason.SHAPE_MISMATCH
    return None


def check_lowering(cluster_nodes) -> RejectionReason | None:
    """Reject here rather than crashing in emit, so the user gets a reason."""
    return None if all(has_lowering(n) for n in cluster_nodes) else RejectionReason.NO_LOWERING


MAX_REDUCTION_BLOCK = 16384  # unmeasured guess; compile_kernel is the real gate


def check_reduction(cluster_nodes, specs) -> RejectionReason | None:
    """Reject what the row-per-program skeleton cannot express."""
    reductions = [n for n in cluster_nodes if classify(n) is OpCategory.REDUCTION]
    if not reductions:
        return None
    if len(reductions) > 1:  # mean+var in one pass needs Welford, v2
        return RejectionReason.UNSUPPORTED_REDUCTION

    node = reductions[0]
    operand = node.args[0] if node.args else None
    spec = specs.get(operand.name) if isinstance(operand, fx.Node) else None
    if spec is None:
        return RejectionReason.UNSUPPORTED_REDUCTION

    axis = reduction_axis(node, len(spec.shape))
    if axis is None or axis != len(spec.shape) - 1:
        return RejectionReason.UNSUPPORTED_REDUCTION

    return RejectionReason.REDUCTION_TOO_LARGE if spec.dims[-1] > MAX_REDUCTION_BLOCK else None


def check_aliasing(cluster_nodes) -> RejectionReason | None:
    """Reject views and in-place mutation, no safe reordering guarantee."""
    for n in cluster_nodes:
        if n.op == "call_method":
            if n.target.endswith("_"):
                return RejectionReason.ALIASING
            if n.target in _VIEW_METHODS:
                return RejectionReason.ALIASING
        elif n.op == "call_function" and n.target in _VIEW_FNS:
            return RejectionReason.ALIASING
    return None
