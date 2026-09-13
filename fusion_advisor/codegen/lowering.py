from __future__ import annotations

import operator

import torch
import torch.nn.functional as F

# target -> fn(operand_exprs...) -> Triton expression string

POINTWISE_RULES: dict[object, callable] = {
    # unary - call_function
    F.relu:         lambda x: f"tl.maximum({x}, 0.0)",
    F.gelu:         lambda x: f"tl.math.gelu({x})",
    F.silu:         lambda x: f"({x}) * tl.sigmoid({x})",
    F.elu:          lambda x: f"tl.where({x} > 0, {x}, tl.math.exp({x}) - 1.0)",
    F.hardswish:    lambda x: f"({x}) * tl.minimum(tl.maximum({x} + 3.0, 0.0), 6.0) / 6.0",
    F.leaky_relu:   lambda x: f"tl.where({x} >= 0, {x}, 0.01 * {x})",
    F.dropout:      lambda x: x,  # inference-only, dropout is identity
    torch.relu:     lambda x: f"tl.maximum({x}, 0.0)",
    torch.sigmoid:  lambda x: f"tl.sigmoid({x})",
    torch.tanh:     lambda x: f"tl.math.tanh({x})",
    torch.exp:      lambda x: f"tl.math.exp({x})",
    torch.log:      lambda x: f"tl.math.log({x})",
    torch.sqrt:     lambda x: f"tl.math.sqrt({x})",
    torch.rsqrt:    lambda x: f"tl.math.rsqrt({x})",
    torch.abs:      lambda x: f"tl.abs({x})",
    torch.neg:      lambda x: f"-({x})",
    torch.erf:      lambda x: f"tl.math.erf({x})",
    torch.sign:     lambda x: f"tl.where({x} > 0, 1.0, tl.where({x} < 0, -1.0, 0.0))",
    operator.neg:   lambda x: f"-({x})",

    # binary - call_function
    operator.add:       lambda x, y: f"({x} + {y})",
    operator.sub:       lambda x, y: f"({x} - {y})",
    operator.mul:       lambda x, y: f"({x} * {y})",
    operator.truediv:   lambda x, y: f"({x} / {y})",
    operator.pow:       lambda x, y: f"tl.math.pow({x}, {y})",
    torch.add:          lambda x, y: f"({x} + {y})",
    torch.sub:          lambda x, y: f"({x} - {y})",
    torch.mul:          lambda x, y: f"({x} * {y})",
    torch.div:          lambda x, y: f"({x} / {y})",
    torch.pow:          lambda x, y: f"tl.math.pow({x}, {y})",
    torch.maximum:      lambda x, y: f"tl.maximum({x}, {y})",
    torch.minimum:      lambda x, y: f"tl.minimum({x}, {y})",
    torch.where:        lambda c, x, y: f"tl.where({c}, {x}, {y})",
}

# call_method targets are strings
POINTWISE_METHOD_RULES: dict[str, callable] = {
    "relu":         lambda x: f"tl.maximum({x}, 0.0)",
    "sigmoid":      lambda x: f"tl.sigmoid({x})",
    "tanh":         lambda x: f"tl.math.tanh({x})",
    "exp":          lambda x: f"tl.math.exp({x})",
    "log":          lambda x: f"tl.math.log({x})",
    "sqrt":         lambda x: f"tl.math.sqrt({x})",
    "rsqrt":        lambda x: f"tl.math.rsqrt({x})",
    "abs":          lambda x: f"tl.abs({x})",
    "neg":          lambda x: f"-({x})",
    "erf":          lambda x: f"tl.math.erf({x})",
    "sign":         lambda x: f"tl.where({x} > 0, 1.0, tl.where({x} < 0, -1.0, 0.0))",
    "add":          lambda x, y: f"({x} + {y})",
    "sub":          lambda x, y: f"({x} - {y})",
    "mul":          lambda x, y: f"({x} * {y})",
    "div":          lambda x, y: f"({x} / {y})",
    "pow":          lambda x, y: f"tl.math.pow({x}, {y})",
    "maximum":      lambda x, y: f"tl.maximum({x}, {y})",
    "minimum":      lambda x, y: f"tl.minimum({x}, {y})",
    "masked_fill":  lambda x, m, v: f"tl.where({m}, {v}, {x})",
}

# target -> fn(operand_expr, axis_str) -> Triton expression string
REDUCTION_RULES: dict[object, callable] = {
    F.softmax:          lambda x, ax: f"tl.softmax({x}, axis={ax})",
    F.log_softmax:      lambda x, ax: f"tl.math.log(tl.softmax({x}, axis={ax}))",
    torch.sum:          lambda x, ax: f"tl.sum({x}, axis={ax})",
    torch.mean:         lambda x, ax: f"tl.sum({x}, axis={ax}) / {x}.shape[{ax}]",
    torch.amax:         lambda x, ax: f"tl.max({x}, axis={ax})",
    torch.amin:         lambda x, ax: f"tl.min({x}, axis={ax})",
}

REDUCTION_METHOD_RULES: dict[str, callable] = {
    "sum":      lambda x, ax: f"tl.sum({x}, axis={ax})",
    "mean":     lambda x, ax: f"tl.sum({x}, axis={ax}) / {x}.shape[{ax}]",
    "softmax":  lambda x, ax: f"tl.softmax({x}, axis={ax})",
    "amax":     lambda x, ax: f"tl.max({x}, axis={ax})",
    "amin":     lambda x, ax: f"tl.min({x}, axis={ax})",
}


def has_lowering(node) -> bool:
    """True if we can emit Triton for this node."""
    if node.op == "call_function":
        return node.target in POINTWISE_RULES or node.target in REDUCTION_RULES
    if node.op == "call_method":
        return node.target in POINTWISE_METHOD_RULES or node.target in REDUCTION_METHOD_RULES
    return False


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

    if node.target in REDUCTION_RULES or (node.op == "call_method" and node.target in REDUCTION_METHOD_RULES):
        return rule(operand_exprs[0], axis or "0")
    return rule(*operand_exprs)
