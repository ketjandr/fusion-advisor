from pathlib import Path

import torch

from fusion_advisor.analysis.cluster import ClusterCategory
from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.analysis.twins import merge_twins
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.loader import load
from fusion_advisor.sourcemap.provenance import MappingQuality, resolve


def test_showcase_example_has_three_exact_clusters():
    path = Path(__file__).parents[1] / "examples" / "example_net.py"
    model = load(str(path)).module
    gm = trace(model)
    specs = propagate(gm, torch.randn(4, 1024))
    clusters, _ = detect(gm, specs)
    ranges = [resolve(cluster, list(gm.graph.nodes)) for cluster in clusters]

    assert len(clusters) == 3
    assert [cluster.category for cluster in clusters] == [
        ClusterCategory.ELEMENTWISE_CHAIN,
        ClusterCategory.ELEMENTWISE_CHAIN,
        ClusterCategory.REDUCTION_BOUNDARY,
    ]
    assert all(source_range.quality is MappingQuality.EXACT for source_range in ranges)


def test_microgpt_layers_share_kernels():
    """Attention plus every residual-add + norm site; the residual is a second output."""
    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    gm = trace(load(str(path)).module)
    specs = propagate(gm, torch.ones(32, 256, dtype=torch.int64))
    clusters, _ = detect(gm, specs)
    clusters = merge_twins(clusters, specs)
    nodes = list(gm.graph.nodes)

    rows = [([n.name for n in c.nodes], c.count, len(c.outputs)) for c in clusters]
    assert rows == [
        (["add", "dropout", "layer_norm"], 1, 2),
        (["mul", "masked_fill", "softmax", "dropout_1"], 2, 1),
        (["dropout_2", "add_1", "layer_norm_1"], 2, 2),
        (["dropout_4", "add_2", "layer_norm_2"], 1, 2),
        (["dropout_8", "add_4", "layer_norm_4"], 1, 1),
    ]
    assert resolve(clusters[1], nodes).quality is MappingQuality.EXACT


def test_microgpt_attention_gets_a_diff():
    """The causal mask is a buffer, named as self.causal_mask in the call."""
    from fusion_advisor.codegen.emit import emit
    from fusion_advisor.sourcemap.diff import call_expression
    from fusion_advisor.sourcemap.provenance import user_variable_names

    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    loaded = load(str(path))
    gm = trace(loaded.module)
    specs = propagate(gm, torch.ones(32, 256, dtype=torch.int64))
    clusters, _ = detect(gm, specs)
    (attention,) = [
        c for c in merge_twins(clusters, specs) if "softmax" in [n.name for n in c.nodes]
    ]
    names = user_variable_names(list(gm.graph.nodes), loaded.source_text)
    kernel = emit(attention, specs)
    attention_name = kernel.name
    call = call_expression(kernel, attention, names)
    assert call == f"{attention_name}(scores, self.causal_mask)"


def test_microgpt_residual_norm_gets_a_diff():
    from fusion_advisor.codegen.emit import emit
    from fusion_advisor.sourcemap.bindings import bind
    from fusion_advisor.sourcemap.diff import build_bound

    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    loaded = load(str(path))
    gm = trace(loaded.module)
    specs = propagate(gm, torch.ones(32, 256, dtype=torch.int64))
    clusters, _ = detect(gm, specs)
    clusters = merge_twins(clusters, specs)
    nodes = list(gm.graph.nodes)
    c = clusters[2]
    rng = resolve(c, nodes)
    diff = build_bound(rng, emit(c, specs), loaded.source_text, bind(c, rng, nodes, loaded.source_text))
    assert diff.added == ["            residual, x = cluster2(x, residual, self.ln_2.weight, self.ln_2.bias)"]


def test_microgpt_rewrite_computes_the_same():
    """Apply every diff, run eager stand-ins for the kernels, compare with the original."""
    from fusion_advisor.codegen.emit import emit
    from fusion_advisor.sourcemap.apply import apply_diffs
    from fusion_advisor.sourcemap.bindings import bind
    from fusion_advisor.sourcemap.diff import build_bound
    from fusion_advisor.validate.harness import extract_subgraph

    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    loaded = load(str(path))
    gm = trace(loaded.module)
    tokens = torch.randint(0, 256, (2, 256))
    specs = propagate(gm, tokens)
    clusters = merge_twins(detect(gm, specs)[0], specs)
    nodes = list(gm.graph.nodes)

    diffs, stand_ins = [], {}
    for c in clusters:
        rng = resolve(c, nodes)
        binding = bind(c, rng, nodes, loaded.source_text) if rng.quality is MappingQuality.EXACT else None
        if binding:
            kernel = emit(c, specs)
            diffs.append(build_bound(rng, kernel, loaded.source_text, binding))
            stand_ins[kernel.name] = extract_subgraph(gm, c)
    assert len(diffs) == 2  # attention and the residual norm

    namespace = {"__name__": "rewritten", **stand_ins}
    source = apply_diffs(loaded.source_text, diffs)
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - our own rewritten example
    rewritten = namespace["MicroGPT"]().eval()
    rewritten.load_state_dict(loaded.module.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(rewritten(tokens), loaded.module(tokens))
