import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")
xfail = pytest.mark.xfail(reason="scaffolding", strict=False)


@gpu
@xfail
def test_generated_kernel_compiles():
    raise NotImplementedError


@gpu
@xfail
def test_generated_kernel_matches_eager():
    """A kernel failing this must never be reported usable."""
    raise NotImplementedError


@gpu
@xfail
def test_speedup_recorded_with_cache_regime():
    """A speedup without its cache regime isn't reproducible."""
    raise NotImplementedError


@gpu
@xfail
def test_large_working_set_shows_dram_savings():
    """Working set > 24 MB L2 - the only regime the bandwidth model describes."""
    raise NotImplementedError


@xfail
def test_cpu_only_degrades_without_claiming_usable():
    raise NotImplementedError
