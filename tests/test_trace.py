import pytest
import torch.fx as fx
import torch.nn as nn

from fusion_advisor.ir.trace import FusionTracer, TracingError, trace
from tests.fixtures import basic


def op_kinds(gm):
    """Map node name -> opcode, skipping plumbing."""
    return {
        n.name: n.op
        for n in gm.graph.nodes
        if n.op not in ("placeholder", "output", "get_attr")
    }


def test_allowlisted_module_is_traced_through():
    gm = trace(basic.ModuleStyle())
    kinds = op_kinds(gm)
    assert "call_module" not in kinds.values(), f"expected all traced through: {kinds}"
    assert len(kinds) >= 5, f"expected >=5 visible ops: {kinds}"


def test_unlisted_module_stays_a_leaf():
    kinds = op_kinds(trace(basic.HasUnlistedModule()))
    assert any(k == "call_module" for k in kinds.values()), f"MHA should be a leaf: {kinds}"
    assert "gelu" in " ".join(kinds), f"GELU should be transparent: {kinds}"


def test_custom_submodule_still_traced_through():
    """Catches `return True` instead of super() - fx traces user modules by default."""
    kinds = op_kinds(trace(basic.CustomSubmodule()))
    assert "call_module" not in kinds.values(), f"user modules are transparent: {kinds}"


def test_is_leaf_module_returns_bool_not_none():
    t = FusionTracer()
    assert t.is_leaf_module(nn.GELU(), "act") is False
    assert isinstance(t.is_leaf_module(nn.MultiheadAttention(8, 2), "attn"), bool)


def test_stack_traces_are_recorded():
    """No provenance, no red/green diff."""
    compute = [n for n in trace(basic.ModuleStyle()).graph.nodes if n.op == "call_function"]
    assert compute
    assert all(n.stack_trace for n in compute)


def test_returns_a_graphmodule_not_a_graph():
    """ShapeProp needs the params bound, which only GraphModule does."""
    gm = trace(basic.ElementwiseChain())
    assert isinstance(gm, fx.GraphModule), f"got {type(gm).__name__}"


def test_untraceable_model_raises_tracing_error():
    with pytest.raises(TracingError):
        trace(basic.Untraceable())


def test_error_message_is_actionable():
    with pytest.raises(TracingError) as exc:
        trace(basic.Untraceable())
    msg = str(exc.value)
    assert "Untraceable" in msg, "name the class"
    assert any(w in msg.lower() for w in ("control flow", "data-dependent")), f"say why: {msg!r}"


def test_original_exception_is_chained():
    """`from e` sets __cause__; bare `raise` only sets __context__."""
    with pytest.raises(TracingError) as exc:
        trace(basic.Untraceable())
    assert exc.value.__cause__ is not None
