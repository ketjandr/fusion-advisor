"""Split composite ops into an opaque core plus a fusable pointwise tail.

F.linear and F.layer_norm trace to one node each, hiding their bias-add and
affine transform from detection. Pulling those out leaves the matmul and the
normalisation opaque while exposing the cheap part as ordinary pointwise ops.
"""

from __future__ import annotations

import operator

import torch.fx as fx
import torch.nn.functional as F


def _arg(node: fx.Node, pos: int, name: str, default=None):
    """Read an argument given either positionally or by keyword."""
    if len(node.args) > pos:
        return node.args[pos]
    return node.kwargs.get(name, default)


def _append(graph: fx.Graph, node: fx.Node, target, operand, stack_trace: str | None) -> fx.Node:
    """Insert `target(node, operand)` and reroute node's users onto it."""
    with graph.inserting_after(node):
        tail = graph.call_function(target, (node, operand))
    # every user except the new node itself, which must keep reading `node`
    node.replace_all_uses_with(tail, delete_user_cb=lambda u: u is not tail)
    tail.stack_trace = stack_trace  # inherited, or the diff loses this line
    return tail


def _split_linear(graph: fx.Graph, node: fx.Node) -> bool:
    bias = _arg(node, 2, "bias")
    if bias is None:
        return False
    _append(graph, node, operator.add, bias, node.stack_trace)
    node.args = (_arg(node, 0, "input"), _arg(node, 1, "weight"), None)
    node.kwargs = {}
    return True


def _split_layer_norm(graph: fx.Graph, node: fx.Node) -> bool:
    weight, bias = _arg(node, 2, "weight"), _arg(node, 3, "bias")
    if weight is None and bias is None:
        return False

    tail = node
    if weight is not None:
        tail = _append(graph, tail, operator.mul, weight, node.stack_trace)
    if bias is not None:
        tail = _append(graph, tail, operator.add, bias, node.stack_trace)

    node.args = (
        _arg(node, 0, "input"),
        _arg(node, 1, "normalized_shape"),
        None,
        None,
        _arg(node, 4, "eps", 1e-5),
    )
    node.kwargs = {}
    return True


SPLITTERS = {F.linear: _split_linear, F.layer_norm: _split_layer_norm}


def decompose(gm: fx.GraphModule) -> fx.GraphModule:
    """Rewrite composite ops in place, exposing their pointwise tails."""
    changed = False
    for node in list(gm.graph.nodes):
        split = SPLITTERS.get(node.target) if node.op == "call_function" else None
        if split is not None:
            changed |= split(gm.graph, node)

    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm
