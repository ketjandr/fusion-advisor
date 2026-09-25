"""Tests for twins.py - one kernel per repeated module, not one per instance."""

from pathlib import Path

import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.analysis.twins import merge_twins
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.sourcemap.diff import build, call_expression
from fusion_advisor.sourcemap.provenance import (
    MappingQuality,
    owner_path,
    resolve,
    user_variable_names,
)
from tests.fixtures import basic

FIXTURE_SRC = Path(basic.__file__).read_text()


def pipeline(model, shape=(4, 64)):
    gm = trace(model)
    specs = propagate(gm, torch.randn(*shape))
    clusters, _ = detect(gm, specs)
    return merge_twins(clusters, specs), specs, list(gm.graph.nodes)


def node_named(nodes, name):
    return next(n for n in nodes if n.name == name)


def test_owner_path_names_the_instance():
    _, _, nodes = pipeline(basic.RepeatedBlocks())
    assert owner_path(node_named(nodes, "relu")) == "blocks.0"
    assert owner_path(node_named(nodes, "relu_1")) == "blocks.1"


def test_owner_path_skips_transparent_modules():
    """F.linear inside nn.Linear belongs to the block, not to blocks.0.fc."""
    _, _, nodes = pipeline(basic.RepeatedBlocks())
    assert owner_path(node_named(nodes, "linear")) == "blocks.0"


def test_owner_path_top_level_is_empty():
    _, _, nodes = pipeline(basic.ElementwiseChain())
    assert owner_path(node_named(nodes, "relu")) == ""


def test_repeated_blocks_merge_into_one():
    clusters, _, _ = pipeline(basic.RepeatedBlocks())
    assert len(clusters) == 1
    assert clusters[0].count == 2


def test_representative_is_the_first_instance():
    """Validation measures the representative, so it should be blocks.0."""
    clusters, _, _ = pipeline(basic.RepeatedBlocks())
    assert [n.name for n in clusters[0].nodes] == ["mul", "relu", "add"]
    assert [n.name for n in clusters[0].instances[0].nodes] == ["mul_1", "relu_1", "add_1"]


def test_different_constant_does_not_merge():
    clusters, _, _ = pipeline(basic.PerLayerScale())
    assert len(clusters) == 2
    assert all(c.count == 1 for c in clusters)


def test_different_shapes_do_not_merge():
    clusters, _, _ = pipeline(basic.VaryingWidthBlocks())
    assert len(clusters) == 2


def test_same_ops_on_different_lines_do_not_merge():
    """ExampleNet-style: two chains of identical ops written separately."""
    clusters, _, _ = pipeline(basic.OpaqueBarrier())
    assert len(clusters) == 2


def test_merge_reindexes():
    clusters, _, _ = pipeline(basic.PerLayerScale())
    assert [c.index for c in clusters] == [0, 1]


def test_no_twins_is_unchanged():
    clusters, _, _ = pipeline(basic.ElementwiseChain())
    assert len(clusters) == 1
    assert clusters[0].count == 1
    assert clusters[0].instances == []


def test_twin_is_not_an_intruder():
    clusters, _, nodes = pipeline(basic.RepeatedBlocks())
    assert resolve(clusters[0], nodes).quality is MappingQuality.EXACT


def test_unmerged_replica_still_blocks_exact():
    """Lines shared with a cluster we did not verify as a twin stay approximate."""
    clusters, _, nodes = pipeline(basic.PerLayerScale())
    assert resolve(clusters[0], nodes).quality is MappingQuality.APPROXIMATE


def test_each_instance_gets_its_variable_name():
    _, _, nodes = pipeline(basic.RepeatedBlocks())
    names = user_variable_names(nodes, FIXTURE_SRC)
    assert names[node_named(nodes, "linear")] == "h"
    assert names[node_named(nodes, "linear_1")] == "h"


def test_repeated_blocks_get_one_diff():
    clusters, specs, nodes = pipeline(basic.RepeatedBlocks())
    c = clusters[0]
    kernel = emit(c, specs)
    call = call_expression(kernel, c, user_variable_names(nodes, FIXTURE_SRC))
    d = build(resolve(c, nodes), kernel, FIXTURE_SRC, call)
    assert d is not None
    assert d.added == ["        return cluster0(h)"]


def test_signature_ignores_node_names():
    """gelu vs gelu_1 is naming, not structure."""
    clusters, specs, _ = pipeline(basic.RepeatedBlocks())
    from fusion_advisor.analysis.twins import signature

    rep, twin = clusters[0], clusters[0].instances[0]
    assert signature(rep, specs) == signature(twin, specs)


def test_merged_kernel_matches_every_instance():
    """The shared kernel's eager reference is the same for both blocks."""
    from fusion_advisor.validate.harness import extract_subgraph

    model = basic.RepeatedBlocks()
    gm = trace(model)
    specs = propagate(gm, torch.randn(4, 64))
    clusters, _ = detect(gm, specs)
    rep = merge_twins(clusters, specs)[0]
    x = torch.randn(4, 64)
    for c in (rep, *rep.instances):
        assert torch.allclose(extract_subgraph(gm, c)(x), torch.relu(x * 2.0) + 1.0)
