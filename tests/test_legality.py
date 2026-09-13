"""Tests for analysis/legality.py - pure graph topology, no shapes needed."""

from fusion_advisor.analysis.cluster import RejectionReason
from fusion_advisor.analysis.legality import check_convexity, check_fan_out, escaping_nodes
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


def test_single_node_cluster_is_trivially_convex():
    c = cluster(basic.ElementwiseChain(), "relu")
    assert check_convexity(c) is None
