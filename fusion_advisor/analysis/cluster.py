from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClusterCategory(Enum):
    ELEMENTWISE_CHAIN = "elementwise-chain"
    REDUCTION_BOUNDARY = "single-reduction-boundary"


class RejectionReason(Enum):
    """Surfaced by --explain-rejections."""

    FAN_OUT = "consumer outside cluster: value must stay materialized"
    NON_CONVEX = "dependency path leaves the cluster and returns: unschedulable"
    SHAPE_MISMATCH = "shapes not provably compatible across cluster"
    ALIASING = "aliasing between cluster members"
    NO_LOWERING = "op has no Triton lowering rule"
    REDUCTION_TOO_LARGE = "reduced axis exceeds one block"
    UNSUPPORTED_REDUCTION = "reduction is not a single reduction over the last axis"


@dataclass
class FusableCluster:
    """A legal, fusable subgraph in topological order."""

    index: int
    category: ClusterCategory
    nodes: list  # fx.Node, topologically ordered

    # Externals include bias/mask/scale operands, not just "the input" - omitting
    # them from the fused side is what inflates a savings estimate.
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)

    # same source, same shapes, other module instances (e.g. blocks.1, blocks.2)
    instances: list[FusableCluster] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Instances sharing this kernel, including this one."""
        return 1 + len(self.instances)


@dataclass
class RejectedCandidate:
    """A candidate that failed legality (never silently dropped)."""

    nodes: list
    reason: RejectionReason
    detail: str = ""
