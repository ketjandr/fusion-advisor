from __future__ import annotations

from collections import deque

import torch.fx as fx

from ..ir.op_registry import OpCategory, classify
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


def _can_absorb(cat: OpCategory) -> bool:
    """True if this op category can join a cluster."""
    return cat in (OpCategory.POINTWISE_UNARY, OpCategory.POINTWISE_BINARY, OpCategory.REDUCTION)


def _absorbable_neighbors(node: fx.Node, absorbable: set[fx.Node]) -> list[fx.Node]:
    """Data-connected absorbable nodes - follow args (backward) and users (forward)."""
    neighbors = []
    for u in node.users:
        if u in absorbable:
            neighbors.append(u)
    for a in node.args:
        if isinstance(a, fx.Node) and a in absorbable:
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

        if len(component) < 2:
            continue

        reason = _first_failure(component, specs)
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
