"""Rewrite a source file in place from EXACT diffs."""

from __future__ import annotations

import ast
from itertools import pairwise


class ApplyError(Exception):
    """Edits could not be applied safely."""


def _import_anchor(source_text: str) -> int:
    """Line to insert an import after: the last top-level import, else the docstring."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError as e:
        raise ApplyError(f"Cannot parse source: {e}") from e

    anchor = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            anchor = node.end_lineno
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and anchor == 0:
            anchor = node.end_lineno  # module docstring
    return anchor


def _overlapping(diffs) -> tuple | None:
    """First pair of diffs sharing a line, or None."""
    ordered = sorted(diffs, key=lambda d: d.start_line)
    for a, b in pairwise(ordered):
        if b.start_line <= a.end_line:
            return (a, b)
    return None


def apply_diffs(source_text: str, diffs, import_line: str | None = None) -> str:
    """Replace each diff's line range, bottom-up so earlier edits don't shift later ones."""
    if not diffs:
        raise ApplyError("Nothing to apply.")

    clash = _overlapping(diffs)
    if clash is not None:
        a, b = clash
        raise ApplyError(
            f"Overlapping edits at lines {a.start_line}-{a.end_line} and "
            f"{b.start_line}-{b.end_line}; refusing to rewrite."
        )

    lines = source_text.splitlines()
    for d in sorted(diffs, key=lambda d: d.start_line, reverse=True):
        if not 1 <= d.start_line <= d.end_line <= len(lines):
            raise ApplyError(f"Diff range {d.start_line}-{d.end_line} is outside the file.")
        if lines[d.start_line - 1 : d.end_line] != d.removed:
            raise ApplyError(
                f"Source at lines {d.start_line}-{d.end_line} changed since analysis."
            )
        lines[d.start_line - 1 : d.end_line] = d.added

    if import_line:
        at = _import_anchor(source_text)
        lines.insert(at, import_line)

    return "\n".join(lines) + "\n"
