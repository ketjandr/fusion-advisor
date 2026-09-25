from __future__ import annotations

import operator

import torch
import torch.nn.functional as F

from ..ir.op_registry import is_training_dropout

_INV_SQRT2 = "0.7071067811865476"

# target -> fn(operand_exprs...) -> Triton expression string

POINTWISE_RULES: dict[object, callable] = {
    F.relu: lambda x: f"tl.maximum({x}, 0.0)",
    F.gelu: lambda x: f"({x} * 0.5 * (1.0 + tl.erf({x} * {_INV_SQRT2})))",
    F.silu: lambda x: f"({x} * tl.sigmoid({x}))",
    F.elu: lambda x: f"tl.where({x} > 0, {x}, tl.exp({x}) - 1.0)",
    F.hardswish: lambda x: f"({x} * tl.minimum(tl.maximum({x} + 3.0, 0.0), 6.0) / 6.0)",
    F.leaky_relu: lambda x: f"tl.where({x} >= 0, {x}, 0.01 * {x})",
    F.dropout: lambda x: x,
    torch.relu: lambda x: f"tl.maximum({x}, 0.0)",
    torch.sigmoid: lambda x: f"tl.sigmoid({x})",
    torch.tanh: lambda x: f"(2.0 * tl.sigmoid(2.0 * {x}) - 1.0)",
    torch.exp: lambda x: f"tl.exp({x})",
    torch.log: lambda x: f"tl.log({x})",
    torch.sqrt: lambda x: f"tl.sqrt({x})",
    torch.rsqrt: lambda x: f"tl.rsqrt({x})",
    torch.abs: lambda x: f"tl.abs({x})",
    torch.neg: lambda x: f"(-{x})",
    torch.erf: lambda x: f"tl.erf({x})",
    torch.sign: lambda x: f"tl.where({x} > 0, 1.0, tl.where({x} < 0, -1.0, 0.0))",
    operator.neg: lambda x: f"(-{x})",
    # binary - call_function
    operator.add: lambda x, y: f"({x} + {y})",
    operator.sub: lambda x, y: f"({x} - {y})",
    operator.mul: lambda x, y: f"({x} * {y})",
    operator.truediv: lambda x, y: f"({x} / {y})",
    operator.pow: lambda x, y: f"({x} ** {y})",
    torch.add: lambda x, y: f"({x} + {y})",
    torch.sub: lambda x, y: f"({x} - {y})",
    torch.mul: lambda x, y: f"({x} * {y})",
    torch.div: lambda x, y: f"({x} / {y})",
    torch.pow: lambda x, y: f"({x} ** {y})",
    torch.maximum: lambda x, y: f"tl.maximum({x}, {y})",
    torch.minimum: lambda x, y: f"tl.minimum({x}, {y})",
    torch.where: lambda c, x, y: f"tl.where({c}, {x}, {y})",
}

# call_method targets are strings
POINTWISE_METHOD_RULES: dict[str, callable] = {
    "relu": lambda x: f"tl.maximum({x}, 0.0)",
    "sigmoid": lambda x: f"tl.sigmoid({x})",
    "tanh": lambda x: f"(2.0 * tl.sigmoid(2.0 * {x}) - 1.0)",
    "exp": lambda x: f"tl.exp({x})",
    "log": lambda x: f"tl.log({x})",
    "sqrt": lambda x: f"tl.sqrt({x})",
    "rsqrt": lambda x: f"tl.rsqrt({x})",
    "abs": lambda x: f"tl.abs({x})",
    "neg": lambda x: f"(-{x})",
    "erf": lambda x: f"tl.erf({x})",
    "clamp": lambda x, lo, hi: f"tl.clamp({x}, {lo}, {hi})",
    "sign": lambda x: f"tl.where({x} > 0, 1.0, tl.where({x} < 0, -1.0, 0.0))",
    "add": lambda x, y: f"({x} + {y})",
    "sub": lambda x, y: f"({x} - {y})",
    "mul": lambda x, y: f"({x} * {y})",
    "div": lambda x, y: f"({x} / {y})",
    "pow": lambda x, y: f"({x} ** {y})",
    "maximum": lambda x, y: f"tl.maximum({x}, {y})",
    "minimum": lambda x, y: f"tl.minimum({x}, {y})",
    "masked_fill": lambda x, m, v: f"tl.where({m}, {v}, {x})",
}

# target -> fn(operand_expr, axis_str) -> Triton expression string
REDUCTION_RULES: dict[object, callable] = {
    F.softmax: lambda x, ax: f"tl.softmax({x}, dim={ax})",
    F.log_softmax: lambda x, ax: f"tl.log(tl.softmax({x}, dim={ax}))",
    torch.sum: lambda x, ax: f"tl.sum({x}, axis={ax})",
    torch.mean: lambda x, ax: f"(tl.sum({x}, axis={ax}) / n_cols)",
    torch.amax: lambda x, ax: f"tl.max({x}, axis={ax})",
    torch.amin: lambda x, ax: f"tl.min({x}, axis={ax})",
}

REDUCTION_METHOD_RULES: dict[str, callable] = {
    "sum": lambda x, ax: f"tl.sum({x}, axis={ax})",
    "mean": lambda x, ax: f"(tl.sum({x}, axis={ax}) / n_cols)",
    "softmax": lambda x, ax: f"tl.softmax({x}, dim={ax})",
    "amax": lambda x, ax: f"tl.max({x}, axis={ax})",
    "amin": lambda x, ax: f"tl.min({x}, axis={ax})",
}


# what a masked-off lane must become so it cannot change the result
_IDENTITY = {
    "sum": "0.0", "mean": "0.0",
    "amax": "-float('inf')", "softmax": "-float('inf')", "log_softmax": "-float('inf')",
    "amin": "float('inf')",
}


def reduction_identity(node) -> str:
    """Value for masked lanes; 0.0 would add exp(0-max) to a softmax sum."""
    name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", "")
    return _IDENTITY.get(name, "0.0")


def has_lowering(node) -> bool:
    """True if we can emit Triton for this node."""
    if is_training_dropout(node):  # the identity rule below would drop the mask
        return False
    if node.op == "call_function":
        return node.target in POINTWISE_RULES or node.target in REDUCTION_RULES
    if node.op == "call_method":
        return node.target in POINTWISE_METHOD_RULES or node.target in REDUCTION_METHOD_RULES
    return False


def is_reduction_target(node) -> bool:
    table = REDUCTION_METHOD_RULES if node.op == "call_method" else REDUCTION_RULES
    return node.target in table


def lower(node, operand_exprs: list[str], axis: str | None = None) -> str:
    """Triton expression for one node given its operands' expressions."""
    if node.op == "call_function":
        rule = POINTWISE_RULES.get(node.target) or REDUCTION_RULES.get(node.target)
    elif node.op == "call_method":
        rule = POINTWISE_METHOD_RULES.get(node.target) or REDUCTION_METHOD_RULES.get(node.target)
    else:
        raise ValueError(f"no lowering for {node.op}:{node.target}")

    if rule is None:
        raise ValueError(f"no lowering rule for {node.target}")

    if is_reduction_target(node):
        return rule(operand_exprs[0], axis or "0")
    return rule(*operand_exprs)
