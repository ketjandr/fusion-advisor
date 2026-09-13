from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata


class ShapePropError(Exception):
    """Model failed to run on the example input."""


@dataclass(frozen=True)
class Dim:
    """One tensor dimension; always statically known."""

    value: int

    def provably_equal(self, other: Dim) -> bool:
        """Never compare dims with `==` outside this module - symbolic backends need this hook."""
        return self.value == other.value


@dataclass(frozen=True)
class TensorSpec:
    """Shape, dtype and layout of one graph edge."""

    shape: tuple[Dim, ...]
    dtype: torch.dtype
    contiguous: bool

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d.value
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(d.value for d in self.shape)

    def same_shape_as(self, other: TensorSpec) -> bool:
        return len(self.shape) == len(other.shape) and all(
            a.provably_equal(b) for a, b in zip(self.shape, other.shape)
        )


def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected = 1
    for size, s in zip(reversed(shape), reversed(stride)):
        if size != 1 and s != expected:  # size-1 dims can hold any stride
            return False
        expected *= size
    return True


def from_meta(meta: TensorMetadata) -> TensorSpec:
    return TensorSpec(
        shape=tuple(Dim(int(s)) for s in meta.shape),
        dtype=meta.dtype,
        contiguous=_is_contiguous(tuple(meta.shape), tuple(meta.stride)),
    )


def propagate(gm: torch.fx.GraphModule, *example_inputs) -> dict[str, TensorSpec]:
    """Run ShapeProp and lift tensor_meta into TensorSpec, keyed by node name."""
    try:
        # catch shape/dtype bugs surface here, not at trace time
        ShapeProp(gm).propagate(*example_inputs)
    except Exception as e:
        shapes = ", ".join(str(tuple(t.shape)) for t in example_inputs if hasattr(t, "shape"))
        root = e.__cause__ or e  # ShapeProp wraps the real error in FX node repr
        raise ShapePropError(f"Model does not run with input shape(s) {shapes}: {root}") from e

    # Nodes producing non-tensors (ints, tuples) get no spec and are simply absent.
    return {
        node.name: from_meta(node.meta["tensor_meta"])
        for node in gm.graph.nodes
        if isinstance(node.meta.get("tensor_meta"), TensorMetadata)
    }
