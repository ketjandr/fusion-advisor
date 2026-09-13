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
from tests.fixtures import basic

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
    """Full pipeline: trace -> shapes -> detect -> emit -> import."""
    gm = trace(model)
    inputs = tuple(torch.randn(*s, device="cuda") for s in input_shapes)
    specs = propagate(gm, *inputs)
    clusters, _ = detect(gm, specs)
    assert len(clusters) == 1, f"expected 1 cluster, got {len(clusters)}"

    kernel = emit(clusters[0], specs)
    return load_kernel(kernel, tmp_path), inputs, kernel


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
    fn, inputs, _ = build(model, *shapes, tmp_path=tmp_path)
    torch.testing.assert_close(fn(*inputs), model(*inputs))


def test_non_multiple_of_block_size(tmp_path):
    """Tail block - mask must suppress the out-of-range lanes."""
    model = basic.ElementwiseChain().cuda()
    fn, inputs, _ = build(model, (3, 101), tmp_path=tmp_path)  # 303 elements
    torch.testing.assert_close(fn(*inputs), model(*inputs))


def test_backward_stub_raises_rather_than_detaching(tmp_path):
    """Forward-only for now, so backward must refuse instead of silently no-op."""
    model = basic.ElementwiseChain().cuda()
    fn, inputs, _ = build(model, (4, 64), tmp_path=tmp_path)
    x = inputs[0].detach().requires_grad_(True)
    assert fn(x).grad_fn is not None  # still wired into the graph
    with pytest.raises(NotImplementedError, match="not generated yet"):
        (fn(x) + x).sum().backward()


@pytest.mark.xfail(reason="broadcast index derivation not implemented", strict=False)
def test_broadcast_bias_matches_eager(tmp_path):
    """[D] bias against [B,S,D] - needs its own offset, not the flat one."""
    model = basic.BroadcastBias().cuda()
    fn, inputs, _ = build(model, (4, 16, 64), tmp_path=tmp_path)
    torch.testing.assert_close(fn(*inputs), model(*inputs))


@pytest.mark.xfail(reason="reduction skeleton body not wired up", strict=False)
def test_reduction_matches_eager(tmp_path):
    model = basic.ReductionBoundary().cuda()
    gm = trace(model)
    x = torch.randn(4, 16, device="cuda")
    mask = torch.randint(0, 2, (4, 16), device="cuda").bool()
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    fn = load_kernel(emit(clusters[0], specs), tmp_path)
    torch.testing.assert_close(fn(x, mask), model(x, mask))
