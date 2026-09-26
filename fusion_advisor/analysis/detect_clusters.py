from __future__ import annotations

from collections import deque

import torch.fx as fx

from ..ir.op_registry import OpCategory, classify, is_identity
from .cluster import ClusterCategory, FusableCluster, RejectedCandidate
from .legality import (
    check_aliasing,
    check_convexity,
    check_fan_out,
    check_lowering,
    check_reduction,
    check_shapes,
    escaping_nodes,
    external_inputs,
)


def _category_for(candidate: list[fx.Node]) -> ClusterCategory:
    """ELEMENTWISE_CHAIN unless any member is a reduction."""
    for n in candidate:
        if classify(n) is OpCategory.REDUCTION:
            return ClusterCategory.REDUCTION_BOUNDARY
    return ClusterCategory.ELEMENTWISE_CHAIN


def _first_failure(candidate: list[fx.Node], specs):
    """Run all legality checks; return the first rejection or None."""
    return (
        check_fan_out(candidate)
        or check_convexity(candidate)
        or check_shapes(candidate, specs)
        or check_aliasing(candidate)
        or check_reduction(candidate, specs)
        or check_lowering(candidate)
    )


def _without_extra_escapes(component: list[fx.Node]) -> list[fx.Node] | None:
    """Drop members that must be materialised anyway, keeping the cluster's result."""
    escaping = escaping_nodes(component)
    if len(escaping) < 2:
        return None
    result = escaping[-1]  # topologically last, so the value the cluster produces
    return [n for n in component if n is result or n not in escaping]


def _real_ops(nodes) -> int:
    """Members that do work; eval dropout alone gives fusion nothing to save."""
    return sum(not is_identity(n) for n in nodes)


def _can_absorb(cat: OpCategory) -> bool:
    """True if this op category can join a cluster."""
    return cat in (OpCategory.POINTWISE_UNARY, OpCategory.POINTWISE_BINARY, OpCategory.REDUCTION)


def _absorbable_neighbors(node: fx.Node, absorbable: set[fx.Node]) -> list[fx.Node]:
    """Data-connected absorbable nodes - follow args (backward) and users (forward)."""
    neighbors = []
    for u in node.users:
        if u in absorbable:
            neighbors.append(u)
    for a in node.all_input_nodes:  # kwargs too, e.g. layer_norm's weight
        if a in absorbable:
            neighbors.append(a)
    return neighbors


def _find_component(seed: fx.Node, absorbable: set[fx.Node], visited: set[fx.Node]) -> list[fx.Node]:
    """BFS from seed through absorbable neighbors, returns the component."""
    queue = deque([seed])
    visited.add(seed)
    component = []
    while queue:
        n = queue.popleft()
        component.append(n)
        for nb in _absorbable_neighbors(n, absorbable):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return component


def detect(gm, specs) -> tuple[list[FusableCluster], list[RejectedCandidate]]:
    """Find fusable clusters via connected components of absorbable nodes."""
    topo_index = {n: i for i, n in enumerate(gm.graph.nodes)}

    absorbable: set[fx.Node] = {n for n in gm.graph.nodes if _can_absorb(classify(n))}

    clusters: list[FusableCluster] = []
    rejected: list[RejectedCandidate] = []
    visited: set[fx.Node] = set()

    for node in gm.graph.nodes:
        if node not in absorbable or node in visited:
            continue

        component = _find_component(node, absorbable, visited)
        component.sort(key=lambda n: topo_index[n])

        if _real_ops(component) < 2:
            continue

        reason = _first_failure(component, specs)
        while reason is not None:  # retry on the part that can still fuse
            smaller = _without_extra_escapes(component)
            if smaller is None or len(smaller) < 2 or len(smaller) == len(component):
                break
            component, reason = smaller, _first_failure(smaller, specs)

        if reason is None and _real_ops(component) < 2:
            continue  # shrinking left a single real op, nothing to fuse
        if reason is None:
            clusters.append(FusableCluster(
                index=len(clusters),
                category=_category_for(component),
                nodes=component,
                inputs=external_inputs(component),
                outputs=escaping_nodes(component),
            ))
        else:
            rejected.append(RejectedCandidate(nodes=component, reason=reason))

    return clusters, rejected
