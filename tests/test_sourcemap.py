"""Tests for sourcemap - fx node provenance and diff safety."""

from pathlib import Path
from types import SimpleNamespace

import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.sourcemap.diff import build
from fusion_advisor.sourcemap.provenance import (
    MappingQuality,
    deepest_user_frame,
    resolve,
    user_frames,
)
from tests.fixtures import basic

FIXTURE_SRC = Path(basic.__file__).read_text()


def pipeline(model, shape):
    """Trace, shape-prop, detect. Returns (clusters, specs, all_nodes)."""
    gm = trace(model)
    specs = propagate(gm, torch.randn(*shape))
    clusters, _ = detect(gm, specs)
    return clusters, specs, list(gm.graph.nodes)


def node_named(all_nodes, name):
    return next(n for n in all_nodes if n.name == name)


def test_flat_functional_model_maps_exactly():
    clusters, _, all_nodes = pipeline(basic.ElementwiseChain(), (4, 64))
    sr = resolve(clusters[0], all_nodes)
    assert sr.quality is MappingQuality.EXACT
    assert sr.file.endswith("basic.py")
    assert sr.start_line < sr.end_line


def test_deepest_user_frame_wins():
    """Nested modules: the first frame is the caller, the last is the author."""
    _, _, all_nodes = pipeline(basic.NestedBlock(), (4, 16, 32))
    gelu = node_named(all_nodes, "gelu")

    frames = user_frames(gelu.stack_trace)
    assert len(frames) > 1, "need a nested call to make this meaningful"
    # outer frame is NestedBlock.forward (the call site), inner is FFN.forward
    assert deepest_user_frame(gelu.stack_trace) == frames[-1]
    assert frames[-1].line != frames[0].line


def test_no_stack_trace_is_unavailable():
    """A node with no user frame has no line to point at."""
    assert deepest_user_frame(None) is None
    assert user_frames(None) == []

    fake = SimpleNamespace(stack_trace=None, target="relu", op="call_method")
    sr = resolve(SimpleNamespace(nodes=[fake]), [fake])
    assert sr.quality is MappingQuality.UNAVAILABLE
    assert sr.start_line is None
    assert sr.fallback_label  # still names the op


def test_torch_internal_frames_are_filtered():
    """site-packages frames belong to torch, not the user."""
    trace_str = (
        'File "/x/.venv/lib/python3.12/site-packages/torch/fx/proxy.py", line 9, in impl\n'
        "    body\n"
    )
    assert deepest_user_frame(trace_str) is None


def test_cluster_spanning_two_functions_is_approximate():
    """NestedBlock authors gelu/mul in FFN.forward but the residual one level up."""
    clusters, _, all_nodes = pipeline(basic.NestedBlock(), (4, 16, 32))
    spanning = [c for c in clusters if len({n.name for n in c.nodes} & {"gelu", "add"}) == 2]
    assert spanning, "expected a cluster crossing the module boundary"
    sr = resolve(spanning[0], all_nodes)
    assert sr.quality is MappingQuality.APPROXIMATE


def test_shared_function_name_does_not_pass_as_one_range():
    """Both methods are called `forward` - name alone must not certify a range.

    The line span covers the class body between them, so a diff here deletes code.
    """
    clusters, _, all_nodes = pipeline(basic.NestedBlock(), (4, 16, 32))
    for c in clusters:
        sr = resolve(c, all_nodes)
        if sr.quality is MappingQuality.EXACT:
            removed = FIXTURE_SRC.splitlines()[sr.start_line - 1 : sr.end_line]
            assert not any(ln.strip().startswith(("def ", "class ")) for ln in removed)


def test_chained_one_liner_does_not_render_diff():
    """A shared line holds non-cluster nodes, so replacing it deletes them."""
    clusters, specs, all_nodes = pipeline(basic.ChainedOneLiner(), (4, 16, 32))
    assert clusters, "expected at least one cluster"
    sr = resolve(clusters[0], all_nodes)
    assert sr.quality is MappingQuality.APPROXIMATE
    assert build(sr, emit(clusters[0], specs), FIXTURE_SRC) is None


def test_diff_replaces_only_the_cluster_lines():
    clusters, specs, all_nodes = pipeline(basic.ElementwiseChain(), (4, 64))
    sr = resolve(clusters[0], all_nodes)
    d = build(sr, emit(clusters[0], specs), FIXTURE_SRC)
    assert d is not None
    assert len(d.removed) == sr.end_line - sr.start_line + 1
    assert d.removed == FIXTURE_SRC.splitlines()[sr.start_line - 1 : sr.end_line]


def test_diff_preserves_assignment_target_and_indent():
    """`x = x + 1.0` -> `x = cluster0(x)`, same indentation."""
    clusters, specs, all_nodes = pipeline(basic.ElementwiseChain(), (4, 64))
    sr = resolve(clusters[0], all_nodes)
    d = build(sr, emit(clusters[0], specs), FIXTURE_SRC)
    (added,) = d.added
    assert added.strip().startswith("x = cluster")
    assert added[: len(added) - len(added.lstrip())] == "        "


def test_diff_preserves_return_statement():
    """The diamond ends in a return, so the replacement must too."""
    clusters, specs, all_nodes = pipeline(basic.ReconvergingDiamond(), (4, 64))
    sr = resolve(clusters[0], all_nodes)
    d = build(sr, emit(clusters[0], specs), FIXTURE_SRC)
    (added,) = d.added
    assert added.strip().startswith("return cluster")


def test_non_exact_never_builds_a_diff():
    """The gate that keeps a confidently wrong edit from reaching the user."""
    for quality in (MappingQuality.APPROXIMATE, MappingQuality.UNAVAILABLE):
        sr = SimpleNamespace(quality=quality, file="f.py", start_line=1, end_line=2)
        assert build(sr, SimpleNamespace(call_expr="k(x)"), "a\nb\nc\n") is None
