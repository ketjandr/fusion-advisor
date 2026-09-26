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


def _is_pointwise(n: fx.Node, absorbable: set[fx.Node]) -> bool:
    return n in absorbable and classify(n) is not OpCategory.REDUCTION


def _grow_around(anchor: fx.Node, absorbable, claimed) -> list[fx.Node]:
    """Pull in producers only the group consumes and consumers only the group feeds."""
    group, work = {anchor}, [anchor]

    def free(n):
        return _is_pointwise(n, absorbable) and n not in claimed and n not in group

    while work:
        n = work.pop()
        # a producer joins once its last user has (checked again as each user joins)
        joins = [p for p in n.all_input_nodes if free(p) and all(u in group for u in p.users)]
        if len(n.users) == 1 and free(next(iter(n.users))):
            joins.append(next(iter(n.users)))
        for m in joins:
            if m not in group:
                group.add(m)
                work.append(m)
    return list(group)


def _legal_or_shrunk(candidate, specs):
    """Candidate after legality, retrying on the part that can still fuse."""
    reason = _first_failure(candidate, specs)
    while reason is not None:
        smaller = _without_extra_escapes(candidate)
        if smaller is None or len(smaller) < 2 or len(smaller) == len(candidate):
            break
        candidate, reason = smaller, _first_failure(smaller, specs)
    return candidate, reason


def detect(gm, specs) -> tuple[list[FusableCluster], list[RejectedCandidate]]:
    """Anchor a cluster on each reduction, then group leftover pointwise ops by connectivity."""
    topo_index = {n: i for i, n in enumerate(gm.graph.nodes)}
    absorbable: set[fx.Node] = {n for n in gm.graph.nodes if _can_absorb(classify(n))}

    found: list[list[fx.Node]] = []
    rejected: list[RejectedCandidate] = []
    claimed: set[fx.Node] = set()

    def consider(candidate):
        candidate = sorted(candidate, key=topo_index.__getitem__)
        if _real_ops(candidate) < 2:
            return
        candidate, reason = _legal_or_shrunk(candidate, specs)
        if reason is not None:
            rejected.append(RejectedCandidate(nodes=candidate, reason=reason))
        elif _real_ops(candidate) >= 2:  # shrinking may leave a single real op
            found.append(candidate)
            claimed.update(candidate)

    # one reduction per kernel, so each reduction seeds its own cluster
    for node in gm.graph.nodes:
        if node in absorbable and classify(node) is OpCategory.REDUCTION:
            consider(_grow_around(node, absorbable, claimed))

    # leftover pointwise ops fuse with whatever they touch
    leftover = {n for n in absorbable if _is_pointwise(n, absorbable) and n not in claimed}
    visited: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node in leftover and node not in visited:
            consider(_find_component(node, leftover, visited))

    found.sort(key=lambda c: topo_index[c[0]])  # report in source order
    clusters = [
        FusableCluster(
            index=i,
            category=_category_for(c),
            nodes=c,
            inputs=external_inputs(c),
            outputs=escaping_nodes(c),
        )
        for i, c in enumerate(found)
    ]
    return clusters, rejected
