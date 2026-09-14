"""Compile, numerics and benchmark; triton is imported only by generated kernels."""

from __future__ import annotations

import functools
import importlib.util
import statistics
import sys
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

from .harness import allocate_inputs, extract_subgraph, rewrite_with_kernel, working_set_bytes

KERNEL_HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"

# Overhead is at most 1/N of a measurement N times the launch floor.
LAUNCH_BOUND_MULTIPLE = 3

# Only used when no measurement is available.
LAUNCH_BOUND_BYTES = 16 << 20
DEFAULT_L2_BYTES = 24 << 20  # RTX 4060 fallback if the device will not say


class CacheRegime(Enum):
    """Which effect a measurement is actually dominated by on this GPU."""

    LAUNCH_BOUND = "launch-bound"  # wall time is overhead, not data movement
    L2_RESIDENT = "l2-resident"  # fits in L2, so the round-trip never reaches DRAM
    DRAM_BOUND = "dram-bound"  # the regime traffic.py describes

    @staticmethod
    def classify(working_set_bytes: int, median_ms: float | None = None) -> CacheRegime:
        """Pass `median_ms` when you have one - bytes alone cannot see the floor."""
        floor = launch_floor_ms()
        if median_ms is not None and floor > 0:
            launch_bound = median_ms < LAUNCH_BOUND_MULTIPLE * floor
        else:  # no GPU to measure on, fall back to the byte guess
            launch_bound = working_set_bytes < LAUNCH_BOUND_BYTES

        if launch_bound:
            return CacheRegime.LAUNCH_BOUND
        if working_set_bytes < l2_bytes():
            return CacheRegime.L2_RESIDENT
        return CacheRegime.DRAM_BOUND


def l2_bytes() -> int:
    """L2 size of the current device, or the 4060's if torch will not report it."""
    if not torch.cuda.is_available():
        return DEFAULT_L2_BYTES
    return getattr(torch.cuda.get_device_properties(0), "L2_cache_size", DEFAULT_L2_BYTES)


@functools.cache
def launch_floor_ms() -> float:
    """Wall time of the smallest possible launch on this machine.

    Measured rather than tuned: the floor is a host property (Python dispatch,
    driver latency) that varies with CPU and driver, not just with the GPU.
    """
    if not torch.cuda.is_available():
        return 0.0
    x = torch.ones(1, device="cuda")
    return time_fn(torch.relu, (x,), 0, warmup=10, iters=50).median_ms


@dataclass
class Timing:
    median_ms: float
    p20_ms: float
    p80_ms: float
    achieved_gbps: float  # bytes_moved / median

@dataclass
class BenchmarkResult:
    regime: CacheRegime
    working_set_bytes: int

    cluster_eager: Timing
    cluster_fused: Timing

    # Always much smaller than cluster-local (Amdahl). Lead with this one.
    model_eager: Timing | None
    model_patched: Timing | None

    peak_gbps: float | None = None
    peak_source: str | None = None

    # Opt-in via --vs-inductor: costs seconds of compile per cluster, and an
    # Inductor column in every run frames this as a competitor to a compiler.
    cluster_inductor: Timing | None = None
    model_inductor: Timing | None = None

    @property
    def cluster_speedup(self) -> float:
        return self.cluster_eager.median_ms / self.cluster_fused.median_ms

    @property
    def model_speedup(self) -> float | None:
        if self.model_eager is None or self.model_patched is None:
            return None
        return self.model_eager.median_ms / self.model_patched.median_ms

    @property
    def vs_inductor(self) -> float | None:
        if self.cluster_inductor is None:
            return None
        return self.cluster_inductor.median_ms / self.cluster_fused.median_ms

    @property
    def pct_of_peak(self) -> float | None:
        if self.peak_gbps is None:
            return None
        return 100.0 * self.cluster_fused.achieved_gbps / self.peak_gbps


@dataclass
class ValidationResult:
    compiled: bool
    numerics_ok: bool | None  # None = not checked (no GPU)
    model_numerics_ok: bool | None
    max_abs_err: float | None
    benchmark: BenchmarkResult | None
    skipped_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Only a compiled AND numerically-verified kernel may be called usable."""
        return self.compiled and self.numerics_ok is True and self.model_numerics_ok is not False


def gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("triton") is not None


def detect_peak_gbps(device: int | None = None) -> float | None:
    """Theoretical DRAM bandwidth from CUDA clock (kHz) and bus width (bits)."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(
        torch.cuda.current_device() if device is None else device
    )
    clock_khz = getattr(props, "memory_clock_rate", None)
    bus_bits = getattr(props, "memory_bus_width", None)
    if not clock_khz or not bus_bits:
        return None
    return 2.0 * clock_khz * 1_000 * (bus_bits / 8) / 1e9


def compile_kernel(kernel, out_dir: Path | None = None):
    """Write the kernel to a file and import it, returning the wrapper.

    Must go through a real file: @triton.jit calls inspect.getsourcelines on the
    kernel, and a function exec'd from a string has no source to read back.
    Importing registers the JIT function; Triton compilation happens lazily on
    the wrapper's first invocation in `validate`.
    """
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="fusion_advisor_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kernel.name}.py"
    path.write_text(f"{KERNEL_HEADER}{kernel.kernel_source}\n\n{kernel.wrapper_source}\n")

    mod_name = f"_fa_kernel_{kernel.name}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, kernel.name)


def time_fn(fn, args, bytes_moved: int, warmup: int = 25, iters: int = 100) -> Timing:
    """Median wall time over `iters` launches, timed with CUDA events.

    No L2 flush between iterations: a small working set stays cached, which is
    the honest thing to report as long as the regime is reported alongside it.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    median = statistics.median(samples)
    return Timing(
        median_ms=median,
        p20_ms=samples[int(0.2 * (len(samples) - 1))],
        p80_ms=samples[int(0.8 * (len(samples) - 1))],
        achieved_gbps=(bytes_moved / (median * 1e-3)) / 1e9 if median > 0 else 0.0,
    )


def time_compiled(module, args, bytes_moved: int) -> Timing | None:
    """torch.compile the eager side and time it; None if it will not compile."""
    try:
        # dynamo raises at call time, not compile time, so time_fn is inside the try
        return time_fn(torch.compile(module), args, bytes_moved)
    except Exception:  # noqa: BLE001 - inductor raises many types; a failure just means no column
        return None


def _model_inputs(gm, specs, device="cuda") -> list[torch.Tensor]:
    """Tensors for the whole model's placeholders."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    return allocate_inputs(
        type("_P", (), {"inputs": placeholders, "outputs": []})(), specs, device
    )


def validate(
    kernel, cluster, gm, specs, *, atol=1e-4, rtol=1e-4, vs_inductor=False,
    peak_gbps=None,
) -> ValidationResult:
    """Compile, check numerics, then benchmark - the first three gate the fourth."""
    if not gpu_available():
        return ValidationResult(False, None, None, None, None, "no CUDA GPU or triton")

    try:
        fused = compile_kernel(kernel)
    except Exception as e:  # noqa: BLE001 - triton raises many types; report, never crash the run
        return ValidationResult(False, None, None, None, None, f"compile failed: {e}")

    # cluster numerics against the real subgraph, never a hand-written reference
    args = allocate_inputs(cluster, specs)
    reference = extract_subgraph(gm, cluster).cuda()
    with torch.no_grad():
        want = reference(*args)
        try:
            # @triton.jit compiles here, not while importing the generated file.
            got = fused(*args)
        except Exception as e:  # noqa: BLE001 - compiler/runtime errors vary by Triton version
            return ValidationResult(
                False,
                None,
                None,
                None,
                None,
                f"JIT compile or first launch failed: {type(e).__name__}: {e}",
            )
    max_err = (got - want).abs().max().item()
    numerics_ok = torch.allclose(got, want, atol=atol, rtol=rtol)
    if not numerics_ok:
        return ValidationResult(True, False, None, max_err, None, "cluster numerics mismatch")

    # whole-model numerics catches substitution bugs a subgraph check cannot
    model_inputs = _model_inputs(gm, specs)
    patched = rewrite_with_kernel(gm, cluster, fused).cuda()
    with torch.no_grad():
        model_ok = torch.allclose(
            patched(*model_inputs), gm.cuda()(*model_inputs), atol=atol, rtol=rtol
        )
    if not model_ok:
        return ValidationResult(True, True, False, max_err, None, "model numerics mismatch")

    nbytes = working_set_bytes(cluster, specs)
    cluster_fused = time_fn(fused, args, nbytes)
    configured_peak = peak_gbps is not None
    peak_gbps = peak_gbps if configured_peak else detect_peak_gbps()
    return ValidationResult(
        compiled=True,
        numerics_ok=True,
        model_numerics_ok=True,
        max_abs_err=max_err,
        benchmark=BenchmarkResult(
            # classify from the measurement, not the byte count
            regime=CacheRegime.classify(nbytes, cluster_fused.median_ms),
            working_set_bytes=nbytes,
            cluster_eager=time_fn(reference, args, nbytes),
            cluster_fused=cluster_fused,
            model_eager=time_fn(gm, model_inputs, nbytes),
            model_patched=time_fn(patched, model_inputs, nbytes),
            peak_gbps=peak_gbps,
            peak_source=("configured" if configured_peak else "detected") if peak_gbps else None,
            cluster_inductor=time_compiled(reference, args, nbytes) if vs_inductor else None,
            model_inductor=time_compiled(gm, model_inputs, nbytes) if vs_inductor else None,
        ),
    )
