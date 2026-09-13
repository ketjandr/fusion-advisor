"""Generated kernels compile and match eager, on real hardware.

Skipped without a GPU. The only tests that execute a kernel - everything in
test_codegen.py stops at string inspection, which can't catch a lowering rule
that emits a call to a function Triton doesn't have.
"""

import pytest
import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.codegen.emit import emit
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic

triton = pytest.importorskip("triton", reason="needs triton")
import triton.language as tl  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def build(model, *input_shapes):
    """Full pipeline: trace -> shapes -> detect -> emit -> exec."""
    gm = trace(model)
    inputs = tuple(torch.randn(*s, device="cuda") for s in input_shapes)
    specs = propagate(gm, *inputs)
    clusters, _ = detect(gm, specs)
    assert len(clusters) == 1, f"expected 1 cluster, got {len(clusters)}"

    kernel = emit(clusters[0], specs)
    ns = {"torch": torch, "triton": triton, "tl": tl}
    exec(kernel.kernel_source, ns)  # noqa: S102
    exec(kernel.wrapper_source, ns)  # noqa: S102
    return ns[kernel.name], inputs, kernel


@pytest.mark.parametrize(
    ("make", "shapes"),
    [
        (basic.ElementwiseChain, [(4, 64)]),
        (basic.ReconvergingDiamond, [(4, 64)]),
        (basic.RepeatedOperand, [(4, 64)]),
    ],
    ids=["chain", "diamond", "repeated"],
)
def test_kernel_matches_eager(make, shapes):
    """The whole point: fused output == eager output."""
    model = make().cuda()
    fn, inputs, _ = build(model, *shapes)
    torch.testing.assert_close(fn(*inputs), model(*inputs))


def test_non_multiple_of_block_size():
    """Tail block - mask must suppress the out-of-range lanes."""
    model = basic.ElementwiseChain().cuda()
    fn, inputs, _ = build(model, (3, 101))  # 303 elements, not a block multiple
    torch.testing.assert_close(fn(*inputs), model(*inputs))


@pytest.mark.xfail(reason="broadcast index derivation not implemented", strict=False)
def test_broadcast_bias_matches_eager():
    """[D] bias against [B,S,D] - needs its own offset, not the flat one."""
    model = basic.BroadcastBias().cuda()
    fn, inputs, _ = build(model, (4, 16, 64))
    torch.testing.assert_close(fn(*inputs), model(*inputs))


@pytest.mark.xfail(reason="reduction skeleton body not wired up", strict=False)
def test_reduction_matches_eager():
    model = basic.ReductionBoundary().cuda()
    gm = trace(model)
    x = torch.randn(4, 16, device="cuda")
    mask = torch.randint(0, 2, (4, 16), device="cuda").bool()
    specs = propagate(gm, x, mask)
    clusters, _ = detect(gm, specs)
    kernel = emit(clusters[0], specs)
    ns = {"torch": torch, "triton": triton, "tl": tl}
    exec(kernel.kernel_source, ns)  # noqa: S102
    exec(kernel.wrapper_source, ns)  # noqa: S102
    torch.testing.assert_close(ns[kernel.name](x, mask), model(x, mask))
