"""Tests for analysis/detect_clusters.py - end-to-end cluster detection."""

import torch

from fusion_advisor.analysis.cluster import ClusterCategory, RejectionReason
from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic


def run(model, *input_shapes):
    """Trace, propagate shapes, detect clusters."""
    gm = trace(model)
    inputs = tuple(torch.randn(*s) for s in input_shapes)
    specs = propagate(gm, *inputs)
    return detect(gm, specs)


def test_elementwise_chain_forms_one_cluster():
    clusters, _ = run(basic.ElementwiseChain(), (4, 64))
    assert len(clusters) == 1
    assert clusters[0].category is ClusterCategory.ELEMENTWISE_CHAIN
    assert len(clusters[0].nodes) >= 2


def test_reconverging_diamond_is_one_cluster():
    clusters, _ = run(basic.ReconvergingDiamond(), (4, 64))
    assert len(clusters) == 1
    node_names = [n.name for n in clusters[0].nodes]
    assert "relu" in node_names
    assert "mul" in node_names
    assert "add" in node_names


def test_opaque_consumer_blocks_fusion():
    """TrueFanOut: matmul makes the cluster non-convex or fan-out."""
    _, rejected = run(basic.TrueFanOut(), (4, 64))
    reasons = [r.reason for r in rejected]
    assert RejectionReason.FAN_OUT in reasons or RejectionReason.NON_CONVEX in reasons


def test_escaping_output_blocks_fusion():
    _, rejected = run(basic.EscapingOutput(), (4, 64))
    reasons = [r.reason for r in rejected]
    assert RejectionReason.FAN_OUT in reasons


def test_repeated_operand_is_legal():
    clusters, _ = run(basic.RepeatedOperand(), (4, 64))
    assert len(clusters) == 1


def test_non_convex_cluster_rejected():
    _, rejected = run(basic.TrueFanOut(), (4, 64))
    reasons = [r.reason for r in rejected]
    assert any(r in reasons for r in (RejectionReason.FAN_OUT, RejectionReason.NON_CONVEX))


def test_matmul_splits_into_two_clusters():
    clusters, _ = run(basic.OpaqueBarrier(), (4, 64))
    assert len(clusters) == 2
    names_0 = {n.name for n in clusters[0].nodes}
    names_1 = {n.name for n in clusters[1].nodes}
    assert "relu" in names_0 or "relu" in names_1
    assert "gelu" in names_0 or "gelu" in names_1


def test_unknown_op_defaults_to_opaque():
    """HasUnlistedModule's attention stays a leaf - barrier, not absorbed."""
    clusters, _ = run(basic.HasUnlistedModule(), (2, 8, 32))
    for c in clusters:
        node_ops = [n.op for n in c.nodes]
        assert "call_module" not in node_ops


def test_broadcast_bias_forms_cluster():
    clusters, _ = run(basic.BroadcastBias(), (4, 16, 64))
    assert len(clusters) == 1
    names = [n.name for n in clusters[0].nodes]
    assert "gelu" in names


def test_reduction_boundary_category():
    gm = trace(basic.ReductionBoundary())
    x = torch.randn(4, 16)
    mask = torch.randint(0, 2, (4, 16)).bool()
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    assert len(clusters) == 1
    assert clusters[0].category is ClusterCategory.REDUCTION_BOUNDARY


def test_view_alias_breaks_connectivity():
    """view is OPAQUE - it splits the component, no cluster forms."""
    clusters, _ = run(basic.ViewAlias(), (4, 64))
    assert len(clusters) == 0


def test_mutation_rejected():
    """add_ is OPAQUE, so the component splits. Whichever check fires first is fine."""
    _, rejected = run(basic.MutationAfterRead(), (4, 64))
    reasons = [r.reason for r in rejected]
    assert any(r in reasons for r in (RejectionReason.FAN_OUT, RejectionReason.ALIASING))


def test_module_style_forms_cluster():
    """Transparent tracing through nn.Module submodules still fuses."""
    clusters, _ = run(basic.ModuleStyle(), (4, 16, 32))
    assert len(clusters) >= 1


def test_single_absorbable_node_discarded():
    """A lone relu between two matmuls is size 1 - not emitted."""
    clusters, _ = run(basic.TrueFanOut(), (4, 64))
    for c in clusters:
        assert len(c.nodes) >= 2


def test_diamond_all_three_nodes_fuse():
    """relu fans into mul and add, but both reconverge - one cluster, no rejections."""
    clusters, rejected = run(basic.ReconvergingDiamond(), (4, 64))
    assert len(clusters) == 1
    assert len(rejected) == 0
    node_names = {n.name for n in clusters[0].nodes}
    assert node_names == {"relu", "mul", "add"}
