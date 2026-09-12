from __future__ import annotations

import operator
from enum import Enum

import torch
import torch.nn.functional as F


class OpCategory(Enum):
    POINTWISE_UNARY = "pointwise_unary"
    POINTWISE_BINARY = "pointwise_binary"
    REDUCTION = "reduction"
    OPAQUE = "opaque"  # matmul, conv, views, and anything unrecognized
    NON_COMPUTE = "non_compute"  # placeholder / output / get_attr


_PLUMBING = ("placeholder", "output", "get_attr")

# call_function targets are callables; call_method targets are plain strings.
# Missing an entry in either table silently splits a cluster, so both matter.

UNARY_FNS = {
    F.relu, F.gelu, F.silu, F.elu, F.hardswish, F.leaky_relu, F.dropout,
    torch.relu, torch.sigmoid, torch.tanh, torch.exp, torch.log,
    torch.sqrt, torch.rsqrt, torch.abs, torch.neg, torch.erf, torch.sign,
    operator.neg,
}

BINARY_FNS = {
    operator.add, operator.sub, operator.mul, operator.truediv, operator.pow,
    torch.add, torch.sub, torch.mul, torch.div, torch.pow,
    torch.maximum, torch.minimum, torch.where,
}

REDUCTION_FNS = {
    F.softmax, F.log_softmax,
    torch.sum, torch.mean, torch.amax, torch.amin, torch.var, torch.logsumexp,
}

UNARY_METHODS = {
    "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt", "abs", "neg",
    "erf", "sign", "clamp", "to", "float", "half",
}

BINARY_METHODS = {
    "add", "sub", "mul", "div", "pow", "maximum", "minimum", "masked_fill",
}

REDUCTION_METHODS = {"sum", "mean", "amax", "amin", "var", "softmax", "logsumexp"}


def classify(node) -> OpCategory:
    """Fusion category of an fx node; anything unrecognized falls back to OPAQUE."""
    if node.op in _PLUMBING:
        return OpCategory.NON_COMPUTE

    if node.op == "call_function":
        if node.target in REDUCTION_FNS:
            return OpCategory.REDUCTION
        if node.target in UNARY_FNS:
            return OpCategory.POINTWISE_UNARY
        if node.target in BINARY_FNS:
            return OpCategory.POINTWISE_BINARY

    elif node.op == "call_method":
        if node.target in REDUCTION_METHODS:
            return OpCategory.REDUCTION
        if node.target in UNARY_METHODS:
            return OpCategory.POINTWISE_UNARY
        if node.target in BINARY_METHODS:
            return OpCategory.POINTWISE_BINARY

    # call_module here means a non-allowlisted leaf
    return OpCategory.OPAQUE


def reduction_axis(node, ndim: int | None = None) -> int | None:
    """Reduction axis as written, normalized if `ndim` given; None if absent or multi-axis."""
    if classify(node) is not OpCategory.REDUCTION:
        return None

    # get axis value from kwargs or positional args
    axis = node.kwargs.get("dim", node.kwargs.get("axis"))
    if axis is None and len(node.args) > 1:
        axis = node.args[1]

    if not isinstance(axis, int) or isinstance(axis, bool):
        return None  # tuple of axes not suported yet

    return axis + ndim if ndim is not None and axis < 0 else axis
