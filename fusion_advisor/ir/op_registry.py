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
    F.softmax, F.log_softmax, F.layer_norm,
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

    if node.target is F.layer_norm:  # normalizes the trailing len(normalized_shape) dims
        shape = call_kwargs(node).get("normalized_shape")
        axis = -1 if isinstance(shape, (tuple, list)) and len(shape) == 1 else None
        if axis is None:
            return None
        return axis + ndim if ndim is not None else axis

    # get axis value from kwargs or positional args
    axis = node.kwargs.get("dim", node.kwargs.get("axis"))
    if axis is None and len(node.args) > 1:
        axis = node.args[1]

    if not isinstance(axis, int) or isinstance(axis, bool):
        return None  # tuple of axes not suported yet

    return axis + ndim if ndim is not None and axis < 0 else axis


def call_kwargs(node) -> dict:
    """A call's arguments by parameter name, however it was spelled; {} if unknown."""
    if node.op != "call_function":
        return {}
    pair = node.normalized_arguments(None, normalize_to_only_use_kwargs=True)
    return dict(pair.kwargs) if pair else {}


def _dropout_training(node) -> bool:
    """F.dropout's `training` flag; torch defaults it to True."""
    if "training" in node.kwargs:
        return bool(node.kwargs["training"])
    return bool(node.args[2]) if len(node.args) > 2 else True


def is_identity(node) -> bool:
    """True for eval-mode dropout, which returns its input tensor untouched."""
    return node.op == "call_function" and node.target is F.dropout and not _dropout_training(node)


def is_training_dropout(node) -> bool:
    """Dropout that actually masks; torch's RNG stream cannot be reproduced in a kernel."""
    return node.op == "call_function" and node.target is F.dropout and _dropout_training(node)
