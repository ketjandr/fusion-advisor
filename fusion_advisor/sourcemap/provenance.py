"""FX node -> original source line, via node.stack_trace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

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
    inside = set(nodes)
    for n in all_nodes:
        if n in inside or n.op in _PLUMBING:
            continue
        f = deepest_user_frame(getattr(n, "stack_trace", None))
        if f and f.path == path and start <= f.line <= end:
            return SourceRange(MappingQuality.APPROXIMATE, path, start, end, _label(nodes))

    return SourceRange(MappingQuality.EXACT, path, start, end, _label(nodes))
