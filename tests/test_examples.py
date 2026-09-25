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
    """Two layers used to give 5 approximate clusters; twins collapse them."""
    path = Path(__file__).parents[1] / "examples" / "microgpt.py"
    gm = trace(load(str(path)).module)
    specs = propagate(gm, torch.ones(32, 256, dtype=torch.int64))
    clusters, _ = detect(gm, specs)
    clusters = merge_twins(clusters, specs)
    nodes = list(gm.graph.nodes)

    assert sorted(c.count for c in clusters) == [1, 2, 2]
    for c in clusters:
        if c.count == 2:  # attention softmax and MLP gelu
            assert resolve(c, nodes).quality is MappingQuality.EXACT
