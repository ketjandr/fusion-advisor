"""Load a user's `mymodel.py` and find the nn.Module to analyze.

`--model mymodel.py` means importing arbitrary user code, which executes it.
Resolution order: `--model-class` if given; else the sole nn.Module subclass
defined in the file; else error listing the candidates found. Never guess
silently when the file defines more than one.

Also records the file's source text and path here -- sourcemap needs the exact
bytes that were traced, not a re-read that may have changed on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoadedModel:
    module: torch.nn.Module
    source_path: str
    source_text: str


def load(path: str, class_name: str | None = None) -> LoadedModel:
    raise NotImplementedError


def build_example_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Materialize the example input that drives shape propagation (PRD 4.2)."""
    raise NotImplementedError
