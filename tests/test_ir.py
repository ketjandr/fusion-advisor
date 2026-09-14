"""Tests for ir/op_registry.py and ir/shapes.py."""

import operator
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fusion_advisor.ir.op_registry import OpCategory, classify, reduction_axis
from fusion_advisor.ir.shapes import Dim, ShapePropError, TensorSpec, propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic


def fake_node(op, target, args=(), kwargs=None):
    return SimpleNamespace(op=op, target=target, args=args, kwargs=kwargs or {})


def test_classifies_pointwise_and_plumbing():
    gm = trace(basic.ElementwiseChain())
    cats = {n.name: classify(n) for n in gm.graph.nodes}
    assert cats["mul"] is OpCategory.POINTWISE_BINARY
    assert cats["relu"] is OpCategory.POINTWISE_UNARY
    assert cats["x"] is OpCategory.NON_COMPUTE


def test_matmul_is_opaque():
    gm = trace(basic.OpaqueBarrier())
    cats = [classify(n) for n in gm.graph.nodes if n.op == "call_function"]
    assert OpCategory.OPAQUE in cats


def test_unknown_target_defaults_to_opaque():
    """Conservative default - an unrecognized op must never be folded in."""
    assert classify(fake_node("call_function", object())) is OpCategory.OPAQUE


def test_call_method_targets_are_strings():
    """x.mean(-1) has target "mean"; torch.mean has the callable. Separate tables."""
    assert classify(fake_node("call_method", "mean")) is OpCategory.REDUCTION
    assert classify(fake_node("call_function", torch.mean)) is OpCategory.REDUCTION


def test_reduction_axis_from_kwarg_and_positional():
    kw = fake_node("call_function", F.softmax, kwargs={"dim": -1})
    pos = fake_node("call_method", "mean", args=(None, -1))
    assert reduction_axis(kw) == -1
    assert reduction_axis(pos) == -1
    assert reduction_axis(kw, ndim=3) == 2


def test_reduction_axis_none_for_non_reductions_and_multi_axis():
    assert reduction_axis(fake_node("call_function", operator.add)) is None
    assert reduction_axis(fake_node("call_function", torch.sum, kwargs={"dim": (1, 2)})) is None


def test_propagate_gives_shapes_and_bytes():
    gm = trace(basic.ElementwiseChain())
    specs = propagate(gm, torch.randn(4, 128))
    assert specs["relu"].dims == (4, 128)
    assert specs["relu"].nbytes == 4 * 128 * 4  # float32
    assert specs["relu"].contiguous


def test_specs_absent_for_non_tensor_nodes():
    """attn returns a tuple, so its tensor_meta isn't a TensorMetadata."""
    gm = trace(basic.HasUnlistedModule())
    specs = propagate(gm, torch.randn(2, 8, 32))
    assert "attn" not in specs
    assert "gelu" in specs


def test_dim_equality_goes_through_provably_equal():
    assert Dim(4).provably_equal(Dim(4))
    assert not Dim(4).provably_equal(Dim(8))


def test_same_shape_as():
    a = TensorSpec((Dim(4), Dim(8)), torch.float32, True)
    b = TensorSpec((Dim(4), Dim(8)), torch.float32, True)
    c = TensorSpec((Dim(8),), torch.float32, True)
    assert a.same_shape_as(b)
    assert not a.same_shape_as(c)


def test_shape_bug_traces_but_fails_at_shapeprop(capsys):
    """Proxies don't check shapes, so the bug only surfaces when the model runs."""
    gm = trace(basic.ShapeBug())  # no error here
    with pytest.raises(ShapePropError) as exc:
        propagate(gm, torch.randn(4, 8))
    assert "(4, 8)" in str(exc.value), "name the input shape"
    assert exc.value.__cause__ is not None
    assert "Traceback" not in capsys.readouterr().err


def test_inplace_and_view_are_opaque():
    """Conservative default is what protects us before check_aliasing exists."""
    for model, unsafe in [(basic.MutationAfterRead(), "add_"), (basic.ViewAlias(), "view")]:
        cats = {n.name: classify(n) for n in trace(model).graph.nodes}
        assert cats[unsafe] is OpCategory.OPAQUE, f"{unsafe} must be a barrier"
