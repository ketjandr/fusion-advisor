"""Red/green diff construction; plain data so JSON output can reuse it."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .provenance import MappingQuality


@dataclass
class ClusterDiff:
    file: str
    start_line: int
    end_line: int
    removed: list[str]
    added: list[str]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _replacement(last_line: str, call_expr: str) -> str:
    """Rebuild the statement around call_expr, reusing the original binding."""
    # fx drops variable names, so the target comes back from the source text
    indent = _indent_of(last_line)
    try:
        stmt = ast.parse(last_line.strip()).body[0]
    except (SyntaxError, IndexError):
        return f"{indent}{call_expr}"

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        return f"{indent}{ast.unparse(stmt.targets[0])} = {call_expr}"
    if isinstance(stmt, ast.Return):
        return f"{indent}return {call_expr}"
    return f"{indent}{call_expr}"


def call_expression(kernel, cluster, names: dict) -> str | None:
    """Kernel call spelled in the user's own variables, or None if one is unknown."""
    args = []
    for n in cluster.inputs:
        if n not in names:
            return None  # fx names like `linear` are not in scope in the user's code
        args.append(names[n])
    return f"{kernel.name}({', '.join(args)})"


def build(source_range, kernel, source_text: str, call_expr: str | None = None) -> ClusterDiff | None:
    """None unless EXACT; a wrong diff is an edit someone applies."""
    if source_range.quality is not MappingQuality.EXACT:
        return None
    if call_expr is None:
        call_expr = kernel.call_expr

    lines = source_text.splitlines()
    start, end = source_range.start_line, source_range.end_line
    if not (1 <= start <= end <= len(lines)):
        return None

    removed = lines[start - 1 : end]
    return ClusterDiff(
        file=source_range.file,
        start_line=start,
        end_line=end,
        removed=removed,
        added=[_replacement(removed[-1], call_expr)],
    )
