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
    """Twins collapse the two layers; eval dropout leaves attention and the final norm."""
    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    gm = trace(load(str(path)).module)
    specs = propagate(gm, torch.ones(32, 256, dtype=torch.int64))
    clusters, _ = detect(gm, specs)
    clusters = merge_twins(clusters, specs)
    nodes = list(gm.graph.nodes)

    attention, final_norm = sorted(clusters, key=lambda c: -c.count)
    assert attention.count == 2
    assert [n.name for n in final_norm.nodes] == ["dropout_8", "add_4", "layer_norm_4"]
    assert [n.name for n in attention.nodes] == ["mul", "masked_fill", "softmax", "dropout_1"]
    assert resolve(attention, nodes).quality is MappingQuality.EXACT


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
    (attention,) = [c for c in merge_twins(clusters, specs) if c.count == 2]
    names = user_variable_names(list(gm.graph.nodes), loaded.source_text)
    kernel = emit(attention, specs)
    attention_name = kernel.name
    call = call_expression(kernel, attention, names)
    assert call == f"{attention_name}(scores, self.causal_mask)"
