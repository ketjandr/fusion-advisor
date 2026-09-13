"""Load a user's .py and find the nn.Module to analyze."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoadedModel:
    module: torch.nn.Module
    source_path: str
    source_text: str  # captured here, not re-read later, so it matches what was traced


def load(path: str, class_name: str | None = None) -> LoadedModel:
    """Import `path` (which executes it) and resolve the model class.

    Order: --model-class if given, else the sole nn.Module subclass in the file,
    else error listing candidates. Never guess when there is more than one.
    """
    raise NotImplementedError


def build_example_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Materialize the example input that drives shape propagation."""
    raise NotImplementedError
