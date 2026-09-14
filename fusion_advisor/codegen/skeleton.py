"""Load -> compute -> store frames the emitter fills, one per cluster category.

Grid and block sizing come from shape-prop output, never hardcoded per model.
The reduction skeleton assumes the whole reduced axis fits in one block;
anything wider needs a two-pass kernel and must be rejected upstream (by legality.py).
"""

from __future__ import annotations

ELEMENTWISE_SKELETON = """\
@triton.jit
def {name}_kernel({params}, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
{body}
"""

REDUCTION_SKELETON = """\
@triton.jit
def {name}_kernel({params}, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    base = row * n_cols + offs
{body}
"""
