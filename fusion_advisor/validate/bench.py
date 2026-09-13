"""Compile, numerics and benchmark; the only package that imports triton.

Order is compile -> cluster numerics -> model numerics -> benchmark, and the
first three gate the fourth absolutely. A kernel that fails numerics has a bug,
not a speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheRegime(Enum):
    """Which effect a measurement is actually dominated by on this GPU."""

    LAUNCH_BOUND = "launch-bound"  # <~1 MB: measuring kernel launch overhead
    L2_RESIDENT = "l2-resident"  # <24 MB on a 4060: L2 absorbs the round-trip
    DRAM_BOUND = "dram-bound"  # >24 MB: the regime traffic.py describes

    @staticmethod
    def classify(working_set_bytes: int) -> CacheRegime:
        raise NotImplementedError


@dataclass
class Timing:
    median_ms: float
    p20_ms: float
    p80_ms: float
    achieved_gbps: float  # bytes_moved / median

    @property
    def pct_of_peak(self) -> float:
        """Against device peak bandwidth; the real quality metric for a memory-bound kernel."""
        raise NotImplementedError


@dataclass
class BenchmarkResult:
    regime: CacheRegime
    working_set_bytes: int

    cluster_eager: Timing
    cluster_fused: Timing

    # Always much smaller than cluster-local (Amdahl). Lead with this one.
    model_eager: Timing | None
    model_patched: Timing | None

    # Opt-in via --vs-inductor: costs seconds of compile per cluster, and an
    # Inductor column in every run frames this as a competitor to a compiler.
    cluster_inductor: Timing | None = None
    model_inductor: Timing | None = None

    @property
    def cluster_speedup(self) -> float:
        raise NotImplementedError

    @property
    def model_speedup(self) -> float | None:
        raise NotImplementedError

    @property
    def vs_inductor(self) -> float | None:
        raise NotImplementedError


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
    raise NotImplementedError


def validate(kernel, cluster, gm, specs) -> ValidationResult:
    raise NotImplementedError
