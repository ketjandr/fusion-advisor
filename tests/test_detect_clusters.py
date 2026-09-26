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


def test_opaque_consumer_is_excluded_not_fused():
    """relu feeds a matmul so it stays materialised; the rest still fuses."""
    clusters, _ = run(basic.TrueFanOut(), (4, 64))
    fused = {n.name for c in clusters for n in c.nodes}
    assert "relu" not in fused
    assert fused == {"mul", "add"}


def test_escaping_output_blocks_fusion():
    _, rejected = run(basic.EscapingOutput(), (4, 64))
    reasons = [r.reason for r in rejected]
    assert RejectionReason.FAN_OUT in reasons


def test_repeated_operand_is_legal():
    clusters, _ = run(basic.RepeatedOperand(), (4, 64))
    assert len(clusters) == 1


def test_non_convex_triple_is_never_formed():
    """{relu, mul, add} would need the kernel to pause for a matmul."""
    clusters, _ = run(basic.TrueFanOut(), (4, 64))
    assert all({n.name for n in c.nodes} != {"relu", "mul", "add"} for c in clusters)


def test_external_mutation_blocks_fusion():
    """add_ mutates a value two members read, and fx does not order that."""
    clusters, rejected = run(basic.MutationAfterRead(), (4, 64))
    assert clusters == []
    assert RejectionReason.ALIASING in [r.reason for r in rejected]


def test_residual_is_dropped_so_the_rest_can_fuse():
    """The transformer shape: a whole-component reject loses the only fusable pair."""
    clusters, _ = run(basic.ResidualReadTwice(), (4, 16, 64))
    assert [{n.name for n in c.nodes} for c in clusters] == [{"gelu", "add_1"}]


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
    clusters, _ = run(basic.ModuleStyle().eval(), (4, 16, 32))
    assert len(clusters) >= 1


def test_training_dropout_is_rejected_not_fused():
    clusters, rejected = run(basic.ModuleStyle(), (4, 16, 32))  # nn.Module defaults to train()
    assert clusters == []
    assert [r.reason for r in rejected] == [RejectionReason.NO_LOWERING]


def test_eval_dropout_alone_is_not_a_cluster():
    """{gelu, dropout} has one real op, so fusing saves nothing."""
    clusters, rejected = run(basic.GeluDropout().eval(), (4, 64))
    assert clusters == []
    assert rejected == []


def test_eval_dropout_does_not_count_as_an_op():
    clusters, _ = run(basic.ModuleStyle().eval(), (4, 16, 32))
    assert {n.name for n in clusters[0].nodes} == {"gelu", "dropout", "add"}


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


# --- layer_norm is one row reduction, fused with its producers ---


def test_add_then_norm_is_one_cluster():
    clusters, _ = run2(basic.AddNorm(), (4, 16, 64), (4, 16, 64))
    (c,) = clusters
    assert [n.name for n in c.nodes] == ["add", "layer_norm"]
    assert c.category is ClusterCategory.REDUCTION_BOUNDARY


def test_keyword_parameters_are_cluster_inputs():
    """weight and bias reach layer_norm by keyword; the kernel must still load them."""
    clusters, _ = run2(basic.AddNorm(), (4, 16, 64), (4, 16, 64))
    assert [n.name for n in clusters[0].inputs] == ["x", "residual", "norm_weight", "norm_bias"]


def test_keyword_producer_joins_the_cluster():
    """A pointwise op feeding layer_norm only by keyword is still a data edge."""
    clusters, _ = run(basic.KeywordProducer(), (4, 16, 64))
    assert [n.name for n in clusters[0].nodes] == ["mul", "layer_norm"]


def test_multi_dim_norm_is_rejected():
    clusters, rejected = run(basic.TwoDimNorm(), (4, 16, 64))
    assert clusters == []
    assert [r.reason for r in rejected] == [RejectionReason.UNSUPPORTED_REDUCTION]


def run2(model, *shapes):
    gm = trace(model)
    specs = propagate(gm, *(torch.randn(*s) for s in shapes))
    return detect(gm, specs)
