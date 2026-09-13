"""Tests for codegen - emit produces valid, structurally correct Triton."""

import ast

import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.codegen.lowering import has_lowering, lower
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic


def first_cluster(model, *input_shapes):
    """Trace, detect, return the first legal cluster + specs."""
    gm = trace(model)
    inputs = tuple(torch.randn(*s) for s in input_shapes)
    specs = propagate(gm, *inputs)
    clusters, _ = detect(gm, specs)
    assert len(clusters) >= 1, "no clusters found"
    return clusters[0], specs


def test_topological_substitution_chains_variables():
    """Each node references variables from earlier nodes, not raw node names."""
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    kernel = emit(cluster, specs)
    lines = kernel.kernel_source.splitlines()
    compute = [ln.strip() for ln in lines if ln.strip().startswith("v")]
    # v0 is loaded, v1/v2/v3 should reference prior v's or constants
    for line in compute[1:]:
        lhs, rhs = line.split(" = ", 1)
        idx = int(lhs[1:])
        # rhs should only reference v0..v{idx-1} or literals, not fx node names
        assert "relu" not in rhs and "mul" not in rhs and "add" not in rhs


def test_emitted_kernel_is_valid_python_syntax():
    """ast.parse catches most emitter bugs without a GPU."""
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    kernel = emit(cluster, specs)
    ast.parse(kernel.kernel_source)
    ast.parse(kernel.wrapper_source)


def test_diamond_emits_valid_kernel():
    """ReconvergingDiamond has a fork - both branches must resolve."""
    cluster, specs = first_cluster(basic.ReconvergingDiamond(), (4, 64))
    kernel = emit(cluster, specs)
    ast.parse(kernel.kernel_source)
    assert "tl.load" in kernel.kernel_source
    assert "tl.store" in kernel.kernel_source


def test_kernel_has_load_and_store():
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    kernel = emit(cluster, specs)
    assert kernel.kernel_source.count("tl.load") >= 1
    assert kernel.kernel_source.count("tl.store") == 1


def test_diamond_single_load_single_store():
    """Diamond has one external input (x) and one escaping output (add)."""
    cluster, specs = first_cluster(basic.ReconvergingDiamond(), (4, 64))
    kernel = emit(cluster, specs)
    assert kernel.kernel_source.count("tl.load") == 1
    assert kernel.kernel_source.count("tl.store") == 1


def test_wrapper_calls_kernel():
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    kernel = emit(cluster, specs)
    assert f"{kernel.name}_kernel[grid]" in kernel.wrapper_source


def test_call_expr_matches_wrapper():
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    kernel = emit(cluster, specs)
    assert kernel.call_expr.startswith(kernel.name + "(")


def test_reduction_uses_reduction_skeleton():
    """ReductionBoundary should use the reduction kernel structure."""
    gm = trace(basic.ReductionBoundary())
    x = torch.randn(4, 16)
    mask = torch.randint(0, 2, (4, 16)).bool()
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    assert len(clusters) >= 1
    kernel = emit(clusters[0], specs)
    assert "n_rows" in kernel.kernel_source
    assert "n_cols" in kernel.kernel_source


def test_has_lowering_matches_registry():
    """Every op in UNARY_FNS/BINARY_FNS should have a lowering rule."""
    from types import SimpleNamespace

    from fusion_advisor.ir.op_registry import BINARY_FNS, UNARY_FNS
    for target in list(UNARY_FNS) + list(BINARY_FNS):
        node = SimpleNamespace(op="call_function", target=target)
        assert has_lowering(node), f"missing lowering for {target}"


def test_lower_produces_expression():
    from types import SimpleNamespace

    import torch.nn.functional as F
    node = SimpleNamespace(op="call_function", target=F.relu, args=[], kwargs={})
    expr = lower(node, ["v0"])
    assert "v0" in expr
    assert "tl.maximum" in expr
