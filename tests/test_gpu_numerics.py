"""Generated kernels compile and match eager, on real hardware.

Skipped without a GPU. The only tests that execute a kernel - everything in
test_codegen.py stops at string inspection, which can't catch a lowering rule
that emits a call to a function Triton doesn't have.
"""

import importlib.util
import sys

import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from fusion_advisor.validate.harness import extract_subgraph
from tests.fixtures import basic


class _Capture(torch.fx.Interpreter):
    """Records every node's value from one real forward pass."""

    def __init__(self, gm):
        super().__init__(gm)
        self.values = {}

    def run_node(self, n):
        self.values[n.name] = out = super().run_node(n)
        return out


triton = pytest.importorskip("triton", reason="needs triton")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"


def load_kernel(kernel, tmp_path):
    """Import the generated module from a real file."""
    path = tmp_path / f"{kernel.name}_gen.py"
    path.write_text(f"{HEADER}{kernel.kernel_source}\n\n{kernel.wrapper_source}\n")

    mod_name = f"fa_gen_{kernel.name}_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, kernel.name)


def build(model, *input_shapes, tmp_path):
    """Full pipeline; returns (fused fn, cluster args, eager reference).

    Cluster args are not the model's args: a cluster can take a parameter
    (a get_attr, like a bias) that never appears in forward()'s signature.
    """
    gm = trace(model)
    model_inputs = tuple(torch.randn(*s, device="cuda") for s in input_shapes)
    specs = propagate(gm, *model_inputs)
    clusters, _ = detect(gm, specs)
    assert len(clusters) == 1, f"expected 1 cluster, got {len(clusters)}"
    cluster = clusters[0]

    capture = _Capture(gm)
    capture.run(*model_inputs)
    args = tuple(capture.values[n.name] for n in cluster.inputs)

    fn = load_kernel(emit(cluster, specs), tmp_path)
    return fn, args, extract_subgraph(gm, cluster).cuda()


@pytest.mark.parametrize(
    ("make", "shapes"),
    [
        (basic.ElementwiseChain, [(4, 64)]),
        (basic.ReconvergingDiamond, [(4, 64)]),
        (basic.RepeatedOperand, [(4, 64)]),
    ],
    ids=["chain", "diamond", "repeated"],
)
def test_kernel_matches_eager(make, shapes, tmp_path):
    """The whole point: fused output == eager output."""
    model = make().cuda()
    fn, args, reference = build(model, *shapes, tmp_path=tmp_path)
    torch.testing.assert_close(fn(*args), reference(*args))


def test_non_multiple_of_block_size(tmp_path):
    """Tail block - mask must suppress the out-of-range lanes."""
    model = basic.ElementwiseChain().cuda()
    fn, args, reference = build(model, (3, 101), tmp_path=tmp_path)  # 303 elements
    torch.testing.assert_close(fn(*args), reference(*args))


def test_backward_stub_raises_rather_than_detaching(tmp_path):
    """Forward-only for now, so backward must refuse instead of silently no-op."""
    model = basic.ElementwiseChain().cuda()
    fn, args, _ = build(model, (4, 64), tmp_path=tmp_path)
    x = args[0].detach().requires_grad_(True)
    assert fn(x).grad_fn is not None  # still wired into the graph
    with pytest.raises(NotImplementedError, match="not generated yet"):
        (fn(x) + x).sum().backward()


def test_launch_floor_is_measured_and_small():
    """The floor is a real per-machine number, not a tuned constant."""
    from fusion_advisor.validate.bench import launch_floor_ms

    floor = launch_floor_ms()
    print(f"\n  launch floor: {floor * 1000:.1f}us")
    assert 0.0 < floor < 1.0  # microseconds, not milliseconds


def test_measurement_beats_the_byte_guess():
    """A huge working set that ran at the floor is still launch-bound."""
    from fusion_advisor.validate.bench import CacheRegime

    huge = 512 << 20
    assert CacheRegime.classify(huge) is CacheRegime.DRAM_BOUND
    assert CacheRegime.classify(huge, median_ms=0.0) is CacheRegime.LAUNCH_BOUND


def test_validate_end_to_end():
    """compile -> cluster numerics -> model numerics -> benchmark, on real hardware."""
    from fusion_advisor.validate.bench import validate

    model = basic.ElementwiseChain().cuda()
    gm = trace(model)
    x = torch.randn(512, 1024, device="cuda")  # 2 MB, past launch-bound
    specs = propagate(gm, x)
    clusters, _ = detect(gm, specs)

    result = validate(emit(clusters[0], specs), clusters[0], gm, specs)
    assert result.usable, result.skipped_reason
    assert result.max_abs_err < 1e-4

    b = result.benchmark
    print(
        f"\n  {b.regime.value}  {b.working_set_bytes / 1e6:.1f} MB"
        f"\n  cluster {b.cluster_speedup:.2f}x   ({b.cluster_fused.achieved_gbps:.0f} GB/s,"
        f" {b.cluster_fused.pct_of_peak:.0f}% of peak)"
        f"\n  model   {b.model_speedup:.2f}x"
    )
    assert b.cluster_fused.median_ms > 0


def test_broadcast_bias_matches_eager(tmp_path):
    """[D] bias against [B,S,D] - needs its own offset, not the flat one."""
    model = basic.BroadcastBias().cuda()
    model.bias.data = torch.randn(64, device="cuda")  # zeros would hide an index bug
    fn, args, reference = build(model, (4, 16, 64), tmp_path=tmp_path)
    torch.testing.assert_close(fn(*args), reference(*args))


def test_broadcast_is_not_accidentally_elementwise(tmp_path):
    """A wrong index still gives the right shape, so check values."""
    model = basic.BroadcastBias().cuda()
    model.bias.data = torch.arange(64, device="cuda").float()  # distinct per column
    fn, args, reference = build(model, (4, 16, 64), tmp_path=tmp_path)
    torch.testing.assert_close(fn(*args), reference(*args))


@pytest.mark.parametrize("cols", [16, 100, 1000], ids=["pow2", "ragged", "wide"])
def test_reduction_matches_eager(tmp_path, cols):
    """Ragged widths leave block lanes that must not join the sum."""
    model = basic.ReductionBoundary().cuda()
    gm = trace(model)
    x = torch.randn(8, cols, device="cuda")
    mask = torch.randint(0, 2, (8, cols), device="cuda").bool()
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    fn = load_kernel(emit(clusters[0], specs), tmp_path)
    torch.testing.assert_close(fn(x, mask), model(x, mask))


@pytest.mark.parametrize("cols", [64, 100], ids=["pow2", "ragged"])
def test_collapsing_reduction_matches_eager(tmp_path, cols):
    """sum(-1) returns one value per row."""
    model = basic.SumReduction().cuda()
    fn, args, reference = build(model, (8, cols), tmp_path=tmp_path)
    torch.testing.assert_close(fn(*args), reference(*args))


def test_every_row_is_reduced_separately(tmp_path):
    """A missing row offset makes every row a copy of row 0."""
    model = basic.ReductionBoundary().cuda()
    gm = trace(model)
    x = torch.randn(8, 32, device="cuda") * 10  # spread out, so rows really differ
    mask = torch.zeros(8, 32, dtype=torch.bool, device="cuda")
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    out = load_kernel(emit(clusters[0], specs), tmp_path)(x, mask)
    torch.testing.assert_close(out, model(x, mask))
    assert not torch.allclose(out[0], out[1])
