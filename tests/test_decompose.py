"""Tests for ir/decompose.py - splitting composite ops without changing results."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.sourcemap.provenance import deepest_user_frame


class LinearAct(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return F.gelu(self.fc(x))


class NormAct(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return F.gelu(self.ln(x))


class NoBias(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, x):
        return F.gelu(self.fc(self.ln(x)))


def fn_names(gm):
    return [n.name for n in gm.graph.nodes if n.op == "call_function"]


def test_linear_bias_becomes_its_own_node():
    assert fn_names(trace(LinearAct())) == ["linear", "add", "gelu"]


def test_layer_norm_affine_becomes_mul_and_add():
    assert fn_names(trace(NormAct())) == ["layer_norm", "mul", "add", "gelu"]


def test_nothing_to_split_is_left_alone():
    """bias=False and elementwise_affine=False have no pointwise tail."""
    assert fn_names(trace(NoBias())) == ["layer_norm", "linear", "gelu"]


def test_matmul_and_normalisation_stay_opaque():
    """Only the cheap tail is exposed; the expensive core is still a barrier."""
    gm = trace(LinearAct())
    linear = next(n for n in gm.graph.nodes if n.name == "linear")
    assert linear.args[2] is None  # bias was pulled out, not inlined


def test_results_are_unchanged():
    torch.manual_seed(0)
    model = NormAct().eval()
    model.ln.weight.data = torch.randn(64)  # a non-trivial affine, so it matters
    model.ln.bias.data = torch.randn(64)
    x = torch.randn(4, 16, 64)
    torch.testing.assert_close(trace(model)(x), model(x))


def test_results_are_unchanged_for_linear():
    torch.manual_seed(0)
    model = LinearAct().eval()
    x = torch.randn(4, 16, 64)
    torch.testing.assert_close(trace(model)(x), model(x))


def test_synthesized_nodes_keep_provenance():
    """Without a stack_trace the diff has no line to point at."""
    gm = trace(LinearAct())
    for node in gm.graph.nodes:
        if node.op == "call_function":
            assert deepest_user_frame(node.stack_trace) is not None, node.name


def test_tail_inherits_the_composite_line():
    """The bias-add belongs to the line that wrote `self.fc(x)`."""
    gm = trace(LinearAct())
    by_name = {n.name: n for n in gm.graph.nodes}
    assert (
        deepest_user_frame(by_name["add"].stack_trace).line
        == deepest_user_frame(by_name["linear"].stack_trace).line
    )


def test_decomposition_creates_clusters_that_did_not_exist():
    """Linear -> activation found nothing before; the bias-add gives it a partner."""
    gm = trace(LinearAct())
    specs = propagate(gm, torch.randn(4, 16, 64))
    clusters, _ = detect(gm, specs)
    assert [[n.name for n in c.nodes] for c in clusters] == [["add", "gelu"]]


def test_layer_norm_tail_fuses_into_one_cluster():
    gm = trace(NormAct())
    specs = propagate(gm, torch.randn(4, 16, 64))
    clusters, _ = detect(gm, specs)
    assert [[n.name for n in c.nodes] for c in clusters] == [["mul", "add", "gelu"]]
