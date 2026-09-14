"""Load a user's .py and find the nn.Module to analyze."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    "int64": torch.int64, "long": torch.int64,
}


class LoadError(Exception):
    """Model file could not be imported or its class resolved."""


@dataclass
class LoadedModel:
    module: nn.Module
    source_path: str
    source_text: str  # captured here, not re-read later, so it matches what was traced


def _import_file(path: Path):
    """Execute the file as a module, or raise LoadError."""
    name = f"_fusion_advisor_target_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoadError(f"Not an importable Python file: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses and typing lookups need it registered
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        del sys.modules[name]
        raise LoadError(f"Importing {path} failed: {type(e).__name__}: {e}") from e
    return mod


def _candidates(mod) -> list[type]:
    """nn.Module subclasses defined in this file, not ones it imported."""
    return [
        cls
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, nn.Module) and cls is not nn.Module and cls.__module__ == mod.__name__
    ]


def _resolve(mod, candidates: list[type], class_name: str | None) -> type:
    """Pick the class to instantiate; never guess between several."""
    if class_name is not None:
        by_name = {c.__name__: c for c in candidates}
        if class_name not in by_name:
            known = ", ".join(sorted(by_name)) or "none"
            raise LoadError(f"No nn.Module named {class_name!r}. Found: {known}")
        return by_name[class_name]

    if not candidates:
        raise LoadError("No nn.Module subclass defined in this file.")
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise LoadError(f"Several models found, pass --model-class. Found: {names}")
    return candidates[0]


def load(path: str, class_name: str | None = None) -> LoadedModel:
    """Import `path` and instantiate its nn.Module in eval mode."""
    p = Path(path).resolve()
    if not p.is_file():
        raise LoadError(f"No such file: {path}")

    mod = _import_file(p)
    cls = _resolve(mod, _candidates(mod), class_name)

    try:
        model = cls()
    except TypeError as e:
        raise LoadError(f"{cls.__name__}() needs constructor arguments: {e}") from e

    # inference-only: dropout must be identity, matching the lowering rule
    return LoadedModel(module=model.eval(), source_path=str(p), source_text=p.read_text())


def parse_dtype(name: str) -> torch.dtype:
    """Map a CLI dtype string onto a torch dtype."""
    key = name.strip().lower().removeprefix("torch.")
    if key not in _DTYPES:
        raise LoadError(f"Unsupported dtype {name!r}. Try: {', '.join(sorted(_DTYPES))}")
    return _DTYPES[key]


def parse_shape(text: str) -> tuple[int, ...]:
    """Parse "4,128" into (4, 128)."""
    try:
        dims = tuple(int(p) for p in text.replace(" ", "").split(",") if p)
    except ValueError as e:
        raise LoadError(f"Bad shape {text!r}, expected comma-separated ints like 4,128") from e
    if not dims or any(d <= 0 for d in dims):
        raise LoadError(f"Bad shape {text!r}, every dimension must be positive")
    return dims


def build_example_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Materialize the example input that drives shape propagation."""
    if dtype.is_floating_point:
        return torch.randn(*shape, dtype=dtype)
    return torch.ones(*shape, dtype=dtype)
