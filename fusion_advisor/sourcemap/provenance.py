"""FX node -> original source line, via node.stack_trace."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum

from ..ir.trace import TRANSPARENT_MODULES

# File "path", line N, in fnname
_FRAME = re.compile(r'^File "(?P<path>.*)", line (?P<line>\d+), in (?P<fn>.*)$')

_PLUMBING = ("placeholder", "output", "get_attr")


class MappingQuality(Enum):
    EXACT = "exact"  # contiguous range in the user's file -> full diff
    APPROXIMATE = "approximate"  # resolved but not safely replaceable -> pointer
    UNAVAILABLE = "unavailable"  # no user frame -> submodule/call name only


@dataclass(frozen=True)
class Frame:
    path: str
    line: int
    fn: str


@dataclass
class SourceRange:
    quality: MappingQuality
    file: str | None
    start_line: int | None
    end_line: int | None
    fallback_label: str | None = None  # e.g. "FeedForwardNetwork.forward"


def _is_user_path(path: str) -> bool:
    """False for torch internals and synthetic names like <eval_with_key>."""
    return "site-packages" not in path and not path.startswith("<")


def user_frames(stack_trace: str | None) -> list[Frame]:
    """Non-torch frames, outermost first."""
    if not stack_trace:
        return []
    out = []
    for raw in stack_trace.strip().splitlines():
        m = _FRAME.match(raw.strip())
        if m and _is_user_path(m["path"]):
            out.append(Frame(m["path"], int(m["line"]), m["fn"]))
    return out


def deepest_user_frame(stack_trace: str | None) -> Frame | None:
    """Where the op was written, or None."""
    frames = user_frames(stack_trace)
    return frames[-1] if frames else None  # [0] is the caller, wrong for nested modules


def _authoring_context(stack_trace: str | None):
    """Function identity that survives two methods both named `forward`."""
    frames = user_frames(stack_trace)
    if not frames:
        return None
    inner = frames[-1]
    return (tuple(frames[:-1]), inner.path, inner.fn)  # call-site chain disambiguates


def owner_path(node) -> str:
    """Qualname of the user module instance that authored `node`, e.g. "blocks.0.mlp"."""
    stack = getattr(node, "meta", {}).get("nn_module_stack", {})  # outermost first
    owners = [
        path for path, (_, cls) in stack.items()
        if not (isinstance(cls, type) and issubclass(cls, TRANSPARENT_MODULES))
    ]
    return owners[-1] if owners else ""


def attribute_expr(node, owner: str) -> str | None:
    """`self.<path>` for a parameter or buffer read inside `owner`, else None."""
    if node.op != "get_attr":
        return None
    prefix = f"{owner}." if owner else ""
    if not node.target.startswith(prefix):
        return None  # read in an outer module and passed down under a local name
    parts = node.target[len(prefix):].split(".")
    return "self" + "".join(f"[{p}]" if p.isdigit() else f".{p}" for p in parts)


def _frames(nodes) -> dict:
    """node -> Frame, skipping nodes with no user frame."""
    out = {}
    for n in nodes:
        f = deepest_user_frame(getattr(n, "stack_trace", None))
        if f is not None:
            out[n] = f
    return out


def _op_name(node) -> str:
    """Readable op name; call_function targets are objects, call_method strings."""
    t = node.target
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def _label(nodes) -> str:
    """Cluster name for when there is no diff to show."""
    head = " -> ".join(_op_name(n) for n in nodes[:3])
    return head + " -> ..." if len(nodes) > 3 else head


def _assign_target(line: str) -> str | None:
    """Single-name assignment target of a source line, else None."""
    try:
        stmt = ast.parse(line.strip()).body[0]
    except (SyntaxError, IndexError):
        return None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        return t.id if isinstance(t, ast.Name) else None
    return None


def user_variable_names(all_nodes, source_text: str) -> dict:
    """fx node -> the variable the user bound it to, only where provable."""
    lines = source_text.splitlines()
    names, by_line = {}, {}

    for n in all_nodes:
        if n.op == "placeholder":
            names[n] = n.name  # a placeholder's fx name is the parameter name
            continue
        f = deepest_user_frame(getattr(n, "stack_trace", None))
        if f and 1 <= f.line <= len(lines):
            # per instance: repeated blocks put one node each on the same line
            by_line.setdefault((f.line, owner_path(n)), []).append(n)

    # only the last node on a line produces that line's value, e.g. h = act(fc(x))
    # binds h to the activation, not to the inner call
    for (line_no, _), line_nodes in by_line.items():
        target = _assign_target(lines[line_no - 1])
        if target:
            names[line_nodes[-1]] = target
    return names


def resolve(cluster, all_nodes) -> SourceRange:
    """EXACT only if one authoring function and no outside node in the range."""
    nodes = cluster.nodes if hasattr(cluster, "nodes") else cluster
    mine = _frames(nodes)

    if not mine:
        return SourceRange(MappingQuality.UNAVAILABLE, None, None, None, _label(nodes))

    path = mine[next(iter(mine))].path
    lines = [f.line for f in mine.values()]
    start, end = min(lines), max(lines)

    # split authorship means the span covers a class body or another method
    if len({_authoring_context(n.stack_trace) for n in mine}) > 1:
        return SourceRange(MappingQuality.APPROXIMATE, path, start, end, _label(nodes))

    # a non-cluster node in range means replacing it deletes that node's code
    # a twin on the same lines is the same code, not an intruder
    twins = getattr(cluster, "instances", [])
    inside = set(nodes).union(*(t.nodes for t in twins))
    for n in all_nodes:
        if n in inside or n.op in _PLUMBING:
            continue
        f = deepest_user_frame(getattr(n, "stack_trace", None))
        if f and f.path == path and start <= f.line <= end:
            return SourceRange(MappingQuality.APPROXIMATE, path, start, end, _label(nodes))

    return SourceRange(MappingQuality.EXACT, path, start, end, _label(nodes))
