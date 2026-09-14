"""Tests for analysis/legality.py - pure graph topology, no shapes needed."""

import torch

from fusion_advisor.analysis.cluster import RejectionReason
from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.analysis.legality import (
    MAX_REDUCTION_BLOCK,
    check_convexity,
    check_fan_out,
    check_reduction,
    escaping_nodes,
)
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic


def cluster(model, *names):
    """Traced nodes picked by name, in graph order."""
    gm = trace(model)
    return [n for n in gm.graph.nodes if n.name in names]


def names(nodes):
    return [n.name for n in nodes]


def test_escaping_nodes_only_the_result():
    """Diamond: relu and mul feed only cluster members, add is the output."""
    c = cluster(basic.ReconvergingDiamond(), "relu", "mul", "add")
    assert names(escaping_nodes(c)) == ["add"]


def test_escaping_nodes_counts_returned_values():
    """EscapingOutput returns h itself, so relu escapes too."""
    c = cluster(basic.EscapingOutput(), "relu", "mul")
    assert set(names(escaping_nodes(c))) == {"relu", "mul"}


def test_escaping_nodes_counts_opaque_consumers():
    """TrueFanOut: relu also feeds the matmul, which is outside."""
    c = cluster(basic.TrueFanOut(), "relu", "mul", "add")
    assert "relu" in names(escaping_nodes(c))


def test_reconverging_diamond_passes_fan_out():
    """The check that `len(users) > 1` gets wrong - residuals look like this."""
    c = cluster(basic.ReconvergingDiamond(), "relu", "mul", "add")
    assert check_fan_out(c) is None


def test_repeated_operand_passes_fan_out():
    """`h + h` - one user, two arg positions."""
    c = cluster(basic.RepeatedOperand(), "relu", "add")
    assert check_fan_out(c) is None


def test_escaping_output_fails_fan_out():
    c = cluster(basic.EscapingOutput(), "relu", "mul")
    assert check_fan_out(c) is RejectionReason.FAN_OUT


def test_true_fan_out_fails():
    c = cluster(basic.TrueFanOut(), "relu", "mul", "add")
    assert check_fan_out(c) is RejectionReason.FAN_OUT


def test_diamond_is_convex():
    c = cluster(basic.ReconvergingDiamond(), "relu", "mul", "add")
    assert check_convexity(c) is None


def test_path_through_matmul_is_not_convex():
    """relu -> matmul -> add leaves and re-enters, so the kernel can't be scheduled."""
    c = cluster(basic.TrueFanOut(), "relu", "mul", "add")
    assert check_convexity(c) is RejectionReason.NON_CONVEX


def test_cluster_without_the_rejoin_is_convex():
    """Dropping `add` removes the re-entry, so {relu, mul} is fine."""
    c = cluster(basic.TrueFanOut(), "relu", "mul")
    assert check_convexity(c) is None


def test_split_clusters_around_matmul_are_convex():
    """OpaqueBarrier's two halves never rejoin."""
    before = cluster(basic.OpaqueBarrier(), "mul", "relu")
    after = cluster(basic.OpaqueBarrier(), "add", "gelu")
    assert check_convexity(before) is None
    assert check_convexity(after) is None


# --- reductions: the row-per-program skeleton only expresses some of them ---


class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.softmax(x * 0.5, dim=self.dim)


class TwoReductions(torch.nn.Module):
    def forward(self, x):
        h = x * 0.5
        return h.sum(-1, keepdim=True) + h.mean(-1, keepdim=True)


def reasons(model, shape):
    gm = trace(model)
    specs = propagate(gm, torch.randn(*shape))
    clusters, rejected = detect(gm, specs)
    return clusters, [r.reason for r in rejected]


def test_reduction_within_one_block_is_legal():
    clusters, _ = reasons(Softmax(), (2, MAX_REDUCTION_BLOCK))
    assert len(clusters) == 1


def test_reduction_wider_than_a_block_is_rejected():
    """The mask would silently reduce only the first block."""
    _, rej = reasons(Softmax(), (2, MAX_REDUCTION_BLOCK * 2))
    assert RejectionReason.REDUCTION_TOO_LARGE in rej


def test_reduction_over_a_non_last_axis_is_rejected():
    """One row per program, so only the trailing axis reduces."""
    _, rej = reasons(Softmax(dim=0), (8, 16))
    assert RejectionReason.UNSUPPORTED_REDUCTION in rej


def test_two_reductions_in_one_cluster_is_rejected():
    """mean+var in one pass needs Welford, v2."""
    _, rej = reasons(TwoReductions(), (8, 16))
    assert RejectionReason.UNSUPPORTED_REDUCTION in rej


def test_pointwise_clusters_are_unaffected():
    c = cluster(basic.ElementwiseChain(), "mul", "relu", "add")
    assert check_reduction(c, {}) is None


def test_single_node_cluster_is_trivially_convex():
    c = cluster(basic.ElementwiseChain(), "relu")
    assert check_convexity(c) is None
