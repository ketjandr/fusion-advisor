"""Tests for bindings.py - naming a cluster's inputs and outputs by replaying assignments."""

from pathlib import Path

import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.sourcemap.bindings import bind
from fusion_advisor.sourcemap.diff import build_bound
from fusion_advisor.sourcemap.provenance import resolve
from tests.fixtures import basic

FIXTURE_SRC = Path(basic.__file__).read_text()


def replay(model, *shapes, pick=0):
    gm = trace(model)
    specs = propagate(gm, *(torch.randn(*s) for s in shapes))
    clusters, _ = detect(gm, specs)
    nodes = list(gm.graph.nodes)
    c = clusters[pick]
    rng = resolve(c, nodes)
    binding = bind(c, rng, nodes, FIXTURE_SRC)
    diff = build_bound(rng, emit(c, specs), FIXTURE_SRC, binding) if binding else None
    return binding, diff


def test_aliased_residual_and_reused_name_are_both_bound():
    """residual = x copies the add's result; x is then overwritten by the norm."""
    _, diff = replay(basic.ResidualBlock(), (4, 16, 64), (4, 16, 64))
    assert diff.added == ["        residual, x = cluster0(x, residual, self.norm.weight, self.norm.bias)"]


def test_output_held_by_two_names_gets_an_alias_line():
    _, diff = replay(basic.AliasedOutput(), (4, 64))
    assert diff.added == ["        h, y = cluster0(x)", "        g = h"]


@pytest.mark.parametrize(
    ("model", "shapes", "expected"),
    [
        (basic.ElementwiseChain(), [(4, 64)], "        x = cluster0(x)"),
        (basic.RMSNorm(), [(8, 64)], "        return cluster0(x, self.weight)"),
    ],
    ids=["reassigned", "returned"],
)
def test_single_output_matches_the_simple_path(model, shapes, expected):
    _, diff = replay(model, *shapes)
    assert diff.added == [expected]


@pytest.mark.parametrize(
    ("model", "shapes"),
    [
        (basic.BranchBeforeCluster(), [(4, 64)]),  # an if the replay does not follow
        (basic.ConstantInRange(), [(4, 64)]),  # removing `scale = 0.5` could lose a value
        (basic.PassedDownBuffer(), [(4, 64)]),  # two parameters, order not recorded
    ],
    ids=["branch", "constant", "two-params"],
)
def test_declines_what_it_cannot_prove(model, shapes):
    binding, _ = replay(model, *shapes)
    assert binding is None
