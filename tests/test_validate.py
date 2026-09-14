import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.validate.bench import (
    CacheRegime,
    Timing,
    ValidationResult,
    gpu_available,
    validate,
)
from fusion_advisor.validate.harness import (
    allocate_inputs,
    extract_subgraph,
    rewrite_with_kernel,
    working_set_bytes,
)
from tests.fixtures import basic


def pipeline(model, *shapes):
    gm = trace(model)
    inputs = tuple(torch.randn(*s) for s in shapes)
    specs = propagate(gm, *inputs)
    clusters, _ = detect(gm, specs)
    return gm, clusters, specs, inputs


# --- extract_subgraph: the eager baseline must come from the real graph ---


def test_subgraph_matches_the_model(tmp_path):
    """ElementwiseChain is one whole-model cluster, so the two must agree exactly."""
    gm, clusters, _, inputs = pipeline(basic.ElementwiseChain(), (4, 64))
    sub = extract_subgraph(gm, clusters[0])
    torch.testing.assert_close(sub(*inputs), gm(*inputs))


def test_subgraph_has_one_placeholder_per_external_input():
    gm, clusters, _, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    sub = extract_subgraph(gm, clusters[0])
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    assert len(placeholders) == len(clusters[0].inputs)


def test_subgraph_contains_only_cluster_ops():
    gm, clusters, _, _ = pipeline(basic.OpaqueBarrier(), (4, 64))
    sub = extract_subgraph(gm, clusters[0])
    compute = [n for n in sub.graph.nodes if n.op in ("call_function", "call_method")]
    assert len(compute) == len(clusters[0].nodes)
    assert not any("matmul" in str(n.target) for n in compute)


def test_diamond_subgraph_preserves_the_fork(tmp_path):
    gm, clusters, _, inputs = pipeline(basic.ReconvergingDiamond(), (4, 64))
    sub = extract_subgraph(gm, clusters[0])
    torch.testing.assert_close(sub(*inputs), gm(*inputs))


# --- rewrite_with_kernel: must not mutate the baseline ---


def fake_wrapper(*args):
    """Stand-in for a compiled kernel; identity on the first argument."""
    return args[0] * 0.0


def test_rewrite_leaves_the_original_untouched():
    """Mutating gm would turn the A/B into a comparison of patched against itself."""
    gm, clusters, _, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    before = [n.name for n in gm.graph.nodes]
    rewrite_with_kernel(gm, clusters[0], fake_wrapper)
    assert [n.name for n in gm.graph.nodes] == before


def test_rewrite_removes_the_cluster_nodes():
    gm, clusters, _, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    patched = rewrite_with_kernel(gm, clusters[0], fake_wrapper)
    remaining = {n.name for n in patched.graph.nodes}
    assert not remaining & {n.name for n in clusters[0].nodes}


def test_rewrite_calls_the_wrapper():
    gm, clusters, _, inputs = pipeline(basic.ElementwiseChain(), (4, 64))
    patched = rewrite_with_kernel(gm, clusters[0], fake_wrapper)
    targets = [n.target for n in patched.graph.nodes if n.op == "call_function"]
    assert fake_wrapper in targets
    torch.testing.assert_close(patched(*inputs), torch.zeros_like(inputs[0]))


def test_rewrite_keeps_ops_outside_the_cluster():
    """OpaqueBarrier's matmul must survive replacing one of its two clusters."""
    gm, clusters, _, _ = pipeline(basic.OpaqueBarrier(), (4, 64))
    patched = rewrite_with_kernel(gm, clusters[0], fake_wrapper)
    assert any("matmul" in str(n.target) for n in patched.graph.nodes)


# --- allocate_inputs ---


def test_allocate_inputs_matches_specs():
    _, clusters, specs, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    tensors = allocate_inputs(clusters[0], specs, device="cpu")
    assert len(tensors) == len(clusters[0].inputs)
    for t, node in zip(tensors, clusters[0].inputs, strict=True):
        assert tuple(t.shape) == specs[node.name].dims
        assert t.dtype is specs[node.name].dtype


def test_working_set_is_inputs_plus_outputs():
    _, clusters, specs, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    c = clusters[0]
    expected = sum(specs[n.name].nbytes for n in c.inputs + c.outputs)
    assert working_set_bytes(c, specs) == expected


# --- regime classification: the honesty guard on every speedup number ---


@pytest.mark.parametrize(
    ("nbytes", "expected"),
    [
        (16 * 1024, CacheRegime.LAUNCH_BOUND),
        (8 << 20, CacheRegime.LAUNCH_BOUND),
        (20 << 20, CacheRegime.L2_RESIDENT),
        (512 << 20, CacheRegime.DRAM_BOUND),
    ],
)
def test_cache_regime(nbytes, expected):
    assert CacheRegime.classify(nbytes) is expected


def test_default_test_shapes_are_launch_bound():
    """4x64 fp32 is 16 KB - any speedup at this size is launch overhead, not bandwidth."""
    _, clusters, specs, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    nbytes = working_set_bytes(clusters[0], specs)
    assert CacheRegime.classify(nbytes) is CacheRegime.LAUNCH_BOUND


def test_falls_back_to_bytes_with_no_floor_to_measure():
    """Off-GPU there is no floor, so a timing must not silently disable the check."""
    if gpu_available():
        pytest.skip("this asserts the no-GPU fallback")
    assert CacheRegime.classify(8 << 20, median_ms=999.0) is CacheRegime.LAUNCH_BOUND
    assert CacheRegime.classify(512 << 20, median_ms=0.0) is CacheRegime.DRAM_BOUND


# --- result arithmetic ---


def timing(ms):
    return Timing(median_ms=ms, p20_ms=ms, p80_ms=ms, achieved_gbps=1.0)


def test_speedups_are_eager_over_fused():
    from fusion_advisor.validate.bench import BenchmarkResult

    r = BenchmarkResult(
        regime=CacheRegime.LAUNCH_BOUND,
        working_set_bytes=1,
        cluster_eager=timing(2.0),
        cluster_fused=timing(1.0),
        model_eager=timing(10.0),
        model_patched=timing(8.0),
    )
    assert r.cluster_speedup == 2.0
    assert r.model_speedup == 1.25
    assert r.vs_inductor is None  # opt-in, absent by default


def test_usable_requires_compiled_and_verified():
    assert not ValidationResult(False, None, None, None, None).usable
    assert not ValidationResult(True, False, None, None, None).usable
    assert not ValidationResult(True, True, False, None, None).usable
    assert ValidationResult(True, True, True, 0.0, None).usable


def test_validate_degrades_without_a_gpu():
    if gpu_available():
        pytest.skip("this asserts the no-GPU path")
    gm, clusters, specs, _ = pipeline(basic.ElementwiseChain(), (4, 64))
    result = validate(None, clusters[0], gm, specs)
    assert result.compiled is False
    assert result.numerics_ok is None
    assert "no CUDA" in result.skipped_reason
