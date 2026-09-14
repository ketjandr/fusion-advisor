"""Tests for codegen - emit produces valid, structurally correct Triton."""

import ast
import math
from types import SimpleNamespace

import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import broadcast_index, emit
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
        _, rhs = line.split(" = ", 1)
        # rhs references earlier v's or literals, never fx node names
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


def reduction_kernel(cols=16):
    gm = trace(basic.ReductionBoundary())
    specs = propagate(gm, torch.randn(4, cols), torch.randint(0, 2, (4, cols)).bool())
    clusters, _ = detect(gm, specs)
    return emit(clusters[0], specs)


def test_reduction_load_offsets_by_row():
    """Without `base` every program reads row 0."""
    src = reduction_kernel().kernel_source
    assert "base = row * n_cols + offs" in src
    assert "tl.load(in_ptr0 + (base)" in src


def test_reduction_uses_block_axis_not_tensor_axis():
    """Axis 1 of the tensor is axis 0 of a one-row block."""
    src = reduction_kernel().kernel_source
    assert "dim=0)" in src
    assert "dim=1)" not in src


def test_tail_lanes_are_neutralised_before_reducing():
    """Leftover lanes would add exp(0 - max) to the softmax sum."""
    src = reduction_kernel().kernel_source
    guard = next(ln for ln in src.splitlines() if "tl.where(mask," in ln)
    assert "-float('inf')" in guard
    assert src.index(guard) < src.index("tl.softmax")


def test_boolean_operand_is_not_loaded_with_inf():
    """-inf does not fit in an int1."""
    src = reduction_kernel().kernel_source
    assert "other=" not in src


def test_negative_infinity_scalar_is_a_defined_literal():
    """A bare `-inf` parses but fails later when Triton resolves the name."""
    gm = trace(basic.NegativeInfinityMask())
    specs = propagate(gm, torch.randn(4, 16), torch.zeros(4, 16, dtype=torch.bool))
    clusters, _ = detect(gm, specs)
    src = emit(clusters[0], specs).kernel_source
    assert '-float("inf")' in src
    assert ", -inf," not in src


def test_reduction_grid_is_one_program_per_row():
    wrapper = reduction_kernel(cols=16).wrapper_source
    assert "[(4,)]" in wrapper
    assert "4, 16, BLOCK_SIZE=16" in wrapper


def test_block_size_is_rounded_up_to_a_power_of_two():
    """tl.arange needs a power of two."""
    assert "BLOCK_SIZE=128" in reduction_kernel(cols=100).wrapper_source


def collapsing_kernel(cols=100):
    gm = trace(basic.SumReduction())
    specs = propagate(gm, torch.randn(8, cols))
    clusters, _ = detect(gm, specs)
    return emit(clusters[0], specs)


def test_collapsing_reduction_stores_one_scalar_per_row():
    """sum(-1) stores a value at `row`, not a block."""
    src = collapsing_kernel().kernel_source
    assert "tl.store(out_ptr0 + row," in src
    assert "mask=mask)" not in src.splitlines()[-2]


def test_collapsing_reduction_allocates_the_reduced_shape():
    """empty_like would give the pre-reduction shape."""
    assert "torch.empty((8,)" in collapsing_kernel().wrapper_source


def test_sum_neutralises_tail_lanes_with_zero():
    """-inf would poison a sum."""
    assert "tl.where(mask, v1, 0.0)" in collapsing_kernel().kernel_source


def test_reduction_kernel_is_valid_python():
    k = reduction_kernel()
    ast.parse(k.kernel_source)
    ast.parse(k.wrapper_source)


def test_has_lowering_matches_registry():
    """Every op in UNARY_FNS/BINARY_FNS should have a lowering rule."""
    from types import SimpleNamespace

    from fusion_advisor.ir.op_registry import BINARY_FNS, UNARY_FNS

    for target in list(UNARY_FNS) + list(BINARY_FNS):
        node = SimpleNamespace(op="call_function", target=target)
        assert has_lowering(node), f"missing lowering for {target}"


def load_wrapper(kernel):
    """exec the generated wrapper with the triton bits stubbed, so it runs on CPU."""

    class FakeKernel:  # supports kernel[grid](...)
        def __getitem__(self, grid):
            return lambda *a, **kw: None

    ns = {
        "torch": torch,
        "triton": SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
        f"{kernel.name}_kernel": FakeKernel(),
    }
    exec(kernel.wrapper_source, ns)  # noqa: S102
    return ns[kernel.name]


def test_generated_output_stays_in_the_autograd_graph():
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    fn = load_wrapper(emit(cluster, specs))
    out = fn(torch.randn(4, 64, requires_grad=True))
    assert out.requires_grad and out.grad_fn is not None


def test_backward_is_a_stub_not_wrong_gradients():
    """Until backward codegen lands this must raise, never return bad grads."""
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    fn = load_wrapper(emit(cluster, specs))
    x = torch.randn(4, 64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="not generated yet"):
        (fn(x) + x).sum().backward()  # residual bypass - the silent case


def test_generated_wrapper_runs_under_no_grad():
    cluster, specs = first_cluster(basic.ElementwiseChain(), (4, 64))
    fn = load_wrapper(emit(cluster, specs))
    with torch.no_grad():
        assert fn(torch.randn(4, 64, requires_grad=True)).shape == (4, 64)


BROADCASTS = [
    ((4, 64), (4, 64)),  # identical, no index math
    ((64,), (4, 16, 64)),  # a [D] bias against [B, S, D]
    ((1, 1, 8, 8), (2, 4, 8, 8)),  # an attention mask
    ((1,), (4, 64)),  # scalar against everything
    ((16, 1), (16, 64)),  # broadcast on the trailing dim, not the leading one
    ((3, 1, 5), (3, 4, 5)),  # a 1 in the middle
    ((1, 7), (2, 3, 7)),  # fewer dims and a leading 1
    ((2, 1, 1), (2, 3, 5)),
]


@pytest.mark.parametrize(("in_dims", "out_dims"), BROADCASTS, ids=str)
def test_broadcast_index_matches_torch(in_dims, out_dims):
    """Compare the emitted index against torch's own broadcast."""
    expr = broadcast_index(in_dims, out_dims)
    x = torch.arange(math.prod(in_dims)).reshape(in_dims)
    offs = torch.arange(math.prod(out_dims))
    got = x.reshape(-1)[eval(expr, {"offs": offs})]
    torch.testing.assert_close(got, x.broadcast_to(out_dims).reshape(-1))


def test_same_shape_needs_no_index_math():
    assert broadcast_index((4, 64), (4, 64)) == "offs"


def test_fully_broadcast_operand_stays_a_block():
    """`0` would load a scalar, not a block."""
    assert broadcast_index((1,), (4, 64)) == "offs * 0"


def test_broadcast_kernel_indexes_the_bias_separately():
    cluster, specs = first_cluster(basic.BroadcastBias(), (4, 16, 64))
    src = emit(cluster, specs).kernel_source
    ast.parse(src)
    assert "% 64" in src, "the [64] bias must not share the flat offset"


def test_wrapper_sizes_output_from_the_right_input():
    """in0 may be the bias, not the activation."""
    cluster, specs = first_cluster(basic.BroadcastBias(), (4, 16, 64))
    kernel = emit(cluster, specs)
    fn = load_wrapper(kernel)
    out = fn(torch.randn(4, 16, 64), torch.zeros(64))
    assert out.shape == (4, 16, 64)


def test_softmax_uses_dim_not_axis():
    """tl.softmax takes `dim`; tl.sum/max/min take `axis`. Not a typo."""
    from types import SimpleNamespace

    import torch.nn.functional as F

    node = SimpleNamespace(op="call_function", target=F.softmax, args=[], kwargs={})
    assert lower(node, ["v0"], "0") == "tl.softmax(v0, dim=0)"


def test_reductions_use_axis():
    from types import SimpleNamespace

    node = SimpleNamespace(op="call_function", target=torch.sum, args=[], kwargs={})
    assert lower(node, ["v0"], "0") == "tl.sum(v0, axis=0)"


def test_lower_produces_expression():
    from types import SimpleNamespace

    import torch.nn.functional as F

    node = SimpleNamespace(op="call_function", target=F.relu, args=[], kwargs={})
    expr = lower(node, ["v0"])
    assert "v0" in expr
    assert "tl.maximum" in expr
