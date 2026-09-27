"""Which source variables hold a cluster's inputs and outputs, found by replaying assignments."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .provenance import attribute_expr, owner_path, user_frames


@dataclass
class Binding:
    args: list[str]  # kernel call arguments, one per cluster input
    targets: list[str]  # one variable per output; empty means the call is returned
    aliases: list[tuple[str, str]] = field(default_factory=list)  # extra `alias = target`


def _function_at(tree, start: int, end: int):
    """Innermost function whose body holds lines start..end."""
    best = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno < start and end <= node.end_lineno
            and (best is None or node.lineno > best.lineno)
        ):
            best = node
    return best


def _in_instance(node, owner: str) -> bool:
    """Created by the module instance `owner` or anything it calls."""
    return owner == "" or owner in node.meta.get("nn_module_stack", {})


def _sites(fn, path: str, owner: str, all_nodes) -> dict | None:
    """node -> the line of `fn` that produced it, for one instance; None if ambiguous."""
    sites = {}
    for n in all_nodes:
        if n.op in ("placeholder", "output") or not _in_instance(n, owner):
            continue
        frames = [
            f for f in user_frames(getattr(n, "stack_trace", None))
            if f.path == path and f.fn == fn.name and fn.lineno <= f.line <= fn.end_lineno
        ]
        if len(frames) > 1:
            return None  # fn calls itself; cannot tell which frame is ours
        if frames:
            sites[n] = frames[0].line
    return sites


def _parameters(fn, owner: str, all_nodes) -> dict:
    """Parameter name -> the node passed in, where that is unambiguous."""
    args = fn.args
    params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if params and params[0] == "self":
        params = params[1:]
    if owner == "":  # the traced root: placeholders carry the parameter names
        by_name = {n.target: n for n in all_nodes if n.op == "placeholder"}
        return {p: by_name[p] for p in params if p in by_name}

    # a submodule: values its nodes read that were made by the caller
    inside = {n for n in all_nodes if n.op != "output" and _in_instance(n, owner)}
    incoming = {a for n in inside for a in n.all_input_nodes if a not in inside}
    if len(params) == 1 and len(incoming) == 1:
        return {params[0]: next(iter(incoming))}
    return {}  # several parameters: which is which is not recorded


def bind(cluster, source_range, all_nodes, source_text: str) -> Binding | None:
    """Replay the authoring method's assignments; None unless every value is named."""
    start, end = source_range.start_line, source_range.end_line
    fn = _function_at(ast.parse(source_text), start, end)
    if fn is None:
        return None
    owner = owner_path(cluster.nodes[0])
    sites = _sites(fn, source_range.file, owner, all_nodes)
    if sites is None:
        return None
    topo = {n: i for i, n in enumerate(all_nodes)}

    env = _parameters(fn, owner, all_nodes)
    at_start: dict | None = None
    assigned: list[str] = []  # names bound inside the range, first assignment first
    returned = None
    for stmt in fn.body:
        if stmt.lineno > end:
            break
        if stmt.lineno < start <= stmt.end_lineno or stmt.lineno <= end < stmt.end_lineno:
            return None  # the range splits a statement
        inside = stmt.lineno >= start
        if inside and at_start is None:
            at_start = dict(env)

        made = [n for n, line in sites.items() if stmt.lineno <= line <= stmt.end_lineno]
        value = max(made, key=topo.__getitem__) if made else None  # h = act(fc(x)) is act

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if value is None and isinstance(stmt.value, ast.Name):
                value = env.get(stmt.value.id)  # a plain alias, e.g. residual = x
            env[name] = value
            if inside and name not in assigned:
                assigned.append(name)
        elif isinstance(stmt, ast.Return) and inside and stmt.end_lineno == end:
            returned = value
        elif isinstance(stmt, ast.Expr) and not inside:
            continue  # docstrings, logging calls
        elif isinstance(stmt, ast.Pass):
            continue
        else:
            return None  # branches, loops, unpacking, +=: not replayed
    if at_start is None:
        return None

    # inputs: a parameter or buffer by path, else whoever holds it when the range starts
    holders: dict = {}
    for name, node in at_start.items():
        if node is not None:
            holders.setdefault(node, name)
    args = []
    for n in cluster.inputs:
        name = attribute_expr(n, owner) if n.op == "get_attr" else None
        name = name or holders.get(n)
        if name is None:
            return None
        args.append(name)

    # outputs: every name bound in the range must end up holding an output or a temporary
    members, outputs = set(cluster.nodes), list(cluster.outputs)
    if any(env[name] not in members for name in assigned):
        return None  # removing that assignment would lose a value the code still has
    if returned is not None:
        if outputs != [returned] or any(env[name] in outputs for name in assigned):
            return None
        return Binding(args, [])

    targets, aliases = [], []
    for out in outputs:
        names = [name for name in assigned if env[name] is out]
        if not names:
            return None
        targets.append(names[0])
        aliases.extend((alias, names[0]) for alias in names[1:])
    return Binding(args, targets, aliases)
