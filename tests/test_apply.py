"""Tests for sourcemap/apply.py - rewriting a file from EXACT diffs."""

import textwrap
from types import SimpleNamespace

import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.sourcemap.apply import ApplyError, apply_diffs
from fusion_advisor.sourcemap.diff import call_expression
from fusion_advisor.sourcemap.provenance import user_variable_names

SRC = "a\nb\nc\nd\ne\nf\n"


def d(start, end, removed, added):
    return SimpleNamespace(start_line=start, end_line=end, removed=removed, added=added)


def test_single_edit():
    out = apply_diffs(SRC, [d(2, 3, ["b", "c"], ["B"])])
    assert out == "a\nB\nd\ne\nf\n"


def test_edits_apply_bottom_up():
    """A shrinking earlier edit must not shift the later one's line numbers."""
    diffs = [d(2, 3, ["b", "c"], ["B"]), d(5, 6, ["e", "f"], ["E"])]
    assert apply_diffs(SRC, diffs) == "a\nB\nd\nE\n"


def test_order_of_input_does_not_matter():
    diffs = [d(5, 6, ["e", "f"], ["E"]), d(2, 3, ["b", "c"], ["B"])]
    assert apply_diffs(SRC, diffs) == "a\nB\nd\nE\n"


def test_overlapping_edits_refused():
    with pytest.raises(ApplyError, match="Overlapping"):
        apply_diffs(SRC, [d(2, 4, ["b", "c", "d"], ["X"]), d(3, 5, ["c", "d", "e"], ["Y"])])


def test_stale_source_refused():
    """Guards against the file changing between analysis and rewrite."""
    with pytest.raises(ApplyError, match="changed since analysis"):
        apply_diffs(SRC, [d(2, 3, ["WRONG", "c"], ["B"])])


def test_range_outside_file_refused():
    with pytest.raises(ApplyError, match="outside the file"):
        apply_diffs(SRC, [d(9, 10, ["x"], ["y"])])


def test_empty_refused():
    with pytest.raises(ApplyError, match="Nothing to apply"):
        apply_diffs(SRC, [])


def test_import_goes_after_last_import():
    src = '"""Doc."""\nimport os\nimport sys\n\nX = 1\n'
    out = apply_diffs(src, [d(5, 5, ["X = 1"], ["X = 2"])], "from k import cluster0")
    assert out.splitlines()[3] == "from k import cluster0"


def test_import_goes_after_docstring_when_no_imports():
    src = '"""Doc."""\nX = 1\n'
    out = apply_diffs(src, [d(2, 2, ["X = 1"], ["X = 2"])], "from k import cluster0")
    assert out.splitlines() == ['"""Doc."""', "from k import cluster0", "X = 2"]


def test_unparseable_source_refused():
    with pytest.raises(ApplyError, match="Cannot parse"):
        apply_diffs("def (\n", [d(1, 1, ["def ("], ["x"])], "import k")


# --- variable-name resolution, which is what makes an applied edit runnable ---

MODEL = """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        h = self.fc(x)
        h = F.gelu(h)
        return h * 0.5
"""


def build_pipeline(src, shape, tmp_path):
    path = tmp_path / "m.py"
    path.write_text(textwrap.dedent(src))
    import importlib.util

    spec = importlib.util.spec_from_file_location("m_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gm = trace(mod.Net().eval())
    specs = propagate(gm, torch.randn(*shape))
    clusters, _ = detect(gm, specs)
    return clusters, specs, list(gm.graph.nodes), path.read_text()


def test_intermediate_gets_its_user_variable(tmp_path):
    """`h = self.fc(x)` binds `h` to the bias-add, the value the line produces."""
    _, _, all_nodes, src = build_pipeline(MODEL, (4, 32), tmp_path)
    names = user_variable_names(all_nodes, src)
    by_name = {n.name: n for n in all_nodes}
    assert names[by_name["add"]] == "h"
    # the bias-less matmul decompose() left behind is not anything the user wrote
    assert by_name["linear"] not in names


def test_placeholder_keeps_its_parameter_name(tmp_path):
    _, _, all_nodes, src = build_pipeline(MODEL, (4, 32), tmp_path)
    names = user_variable_names(all_nodes, src)
    x = next(n for n in all_nodes if n.op == "placeholder")
    assert names[x] == "x"


FUNCTIONAL = """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def forward(self, x):
        h = x * 2.0
        h = F.gelu(h)
        return h * 0.5
"""


def test_call_expression_uses_user_names(tmp_path):
    clusters, specs, all_nodes, src = build_pipeline(FUNCTIONAL, (4, 32), tmp_path)
    names = user_variable_names(all_nodes, src)
    expr = call_expression(emit(clusters[0], specs), clusters[0], names)
    assert expr == "cluster0(x)"


def test_decomposed_cluster_declines_a_diff(tmp_path):
    """decompose() trades a diff for a cluster: its input is the bias-less matmul,
    which the user never wrote and so cannot be named in a replacement line."""
    clusters, specs, all_nodes, src = build_pipeline(MODEL, (4, 32), tmp_path)
    names = user_variable_names(all_nodes, src)
    assert [n.name for n in clusters[0].inputs] == ["linear", "fc_bias"]
    assert call_expression(emit(clusters[0], specs), clusters[0], names) is None


def test_call_expression_declines_unknown_name():
    """No resolved variable means the emitted call would not compile."""

    class FakeNode:  # hashable, like a real fx.Node
        name = "linear"

    kernel = SimpleNamespace(name="cluster0")
    cluster = SimpleNamespace(inputs=[FakeNode()])
    assert call_expression(kernel, cluster, {}) is None


def test_chained_line_binds_only_the_outer_call(tmp_path):
    """`h = act(fc(x))` binds h to the activation, never to the inner matmul."""
    chained = MODEL.replace(
        "        h = self.fc(x)\n        h = F.gelu(h)\n", "        h = F.gelu(self.fc(x))\n"
    )
    _, _, all_nodes, src = build_pipeline(chained, (4, 32), tmp_path)
    names = user_variable_names(all_nodes, src)
    linear = next(n for n in all_nodes if n.name == "linear")
    gelu = next(n for n in all_nodes if n.name == "gelu")
    assert linear not in names
    assert names[gelu] == "h"
