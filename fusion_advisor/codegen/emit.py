from __future__ import annotations

from dataclasses import dataclass

import torch.fx as fx

from fusion_advisor.ir.op_registry import OpCategory, classify, reduction_axis

from ..analysis.cluster import ClusterCategory, FusableCluster
from .lowering import lower
from .skeleton import ELEMENTWISE_SKELETON, REDUCTION_SKELETON


@dataclass
class GeneratedKernel:
    name: str
    kernel_source: str
    wrapper_source: str
    call_expr: str  # what the diff shows, e.g. "cluster0(x)"


def _resolve_arg(arg, node_to_var: dict[fx.Node, str]) -> str:
    """Turn an fx arg into a Triton expression fragment."""
    if isinstance(arg, fx.Node):
        return node_to_var[arg]
    if isinstance(arg, (int, float)):
        return repr(arg)
    if isinstance(arg, bool):
        return "True" if arg else "False"
    return repr(arg)


def _indent(lines: list[str], spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln for ln in lines)


def broadcast_index(in_dims: tuple[int, ...], out_dims: tuple[int, ...]) -> str:
    """Flat index into an operand, given the flat output index `offs`."""
    if tuple(in_dims) == tuple(out_dims):
        return "offs"

    pad = len(out_dims) - len(in_dims)  # right-align, numpy style
    terms, out_div, in_stride = [], 1, 1
    for i in reversed(range(len(out_dims))):
        j = i - pad
        size = in_dims[j] if j >= 0 else 1
        if size != 1:  # a size-1 or absent dim contributes nothing
            coord = "offs" if out_div == 1 else f"offs // {out_div}"
            coord = f"({coord}) % {out_dims[i]}"
            terms.append(coord if in_stride == 1 else f"({coord}) * {in_stride}")
            in_stride *= size
        out_div *= out_dims[i]

    return " + ".join(reversed(terms)) or "offs * 0"  # all broadcast, still a block


def emit(cluster: FusableCluster, specs: dict) -> GeneratedKernel:
    """Emit a Triton kernel + wrapper for one cluster."""
    nodes = cluster.nodes
    inputs, escaping = cluster.inputs, cluster.outputs
    name = f"cluster{cluster.index}"

    # assign variable names
    node_to_var: dict[fx.Node, str] = {}
    var_count = 0

    out_dims = specs[escaping[0].name].dims

    # tl.load lines, one per external input
    load_lines: list[str] = []
    in_ptrs: list[str] = []
    for i, inp in enumerate(inputs):
        ptr = f"in_ptr{i}"
        var = f"v{var_count}"
        in_ptrs.append(ptr)
        node_to_var[inp] = var
        index = broadcast_index(specs[inp.name].dims, out_dims)
        load_lines.append(f"{var} = tl.load({ptr} + ({index}), mask=mask)")
        var_count += 1

    # walk cluster nodes in order, resolve args, lower
    compute_lines: list[str] = []
    for node in nodes:
        operand_exprs = []

        # we want to extract dim separately for operands for reduction nodes
        if classify(node) == OpCategory.REDUCTION:
            # only the tensor input is an operand
            operand_exprs.append(_resolve_arg(node.args[0], node_to_var))

            # extract dim/axis separately
            spec = specs.get(node.args[0].name) if isinstance(node.args[0], fx.Node) else None
            ndim = len(spec.shape) if spec else None
            axis = reduction_axis(node, ndim)
            assert axis is not None  # legality pass should've proved that axis is accepted

            expr = lower(node, operand_exprs, str(axis))
        else:  # normal case
            operand_exprs = [_resolve_arg(arg, node_to_var) for arg in node.args]
            expr = lower(node, operand_exprs)

        var = f"v{var_count}"
        compute_lines.append(f"{var} = {expr}")
        var_count += 1
        node_to_var[node] = var  # add this so downstream nodes can use the expr node

    # tl.store lines (one per escaping node)
    store_lines: list[str] = []
    out_ptrs: list[str] = []
    for i, esc in enumerate(escaping):
        ptr = f"out_ptr{i}"
        out_ptrs.append(ptr)
        store_lines.append(f"tl.store({ptr} + offs, {node_to_var[esc]}, mask=mask)")

    # assemble kernel
    params = ", ".join(in_ptrs + out_ptrs)
    body = _indent(load_lines + compute_lines + store_lines)
    skeleton = REDUCTION_SKELETON if cluster.category is ClusterCategory.REDUCTION_BOUNDARY else ELEMENTWISE_SKELETON
    kernel_src = skeleton.format(name=name, params=params, body=body)

    # assemble wrapper
    in_args = [f"in{i}" for i in range(len(in_ptrs))]
    args = ", ".join(in_args)
    like = in_args[next((i for i, n in enumerate(inputs) if specs[n.name].dims == out_dims), 0)]
    wrapper_lines = [
        f"class _{name}(torch.autograd.Function):",
        "    @staticmethod",
        f"    def forward(ctx, {args}):",
        f"        out0 = torch.empty_like({like})",
        f"        n = {like}.numel()",
        "        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)",
        f"        {name}_kernel[grid]({args}, out0, n, BLOCK_SIZE=1024)",
        f"        ctx.save_for_backward({args})  # what a backward kernel will need",
        "        return out0",
        "",
        "    @staticmethod",
        "    def backward(ctx, grad_out):",
        "        # TODO: a derivative rule per op, chained in reverse over saved_tensors",
        f'        raise NotImplementedError("{name}: backward not generated yet")',
        "",
        "",
        f"def {name}({args}):",
        f"    return _{name}.apply({args})",
    ]
    wrapper_src = "\n".join(wrapper_lines)

    # diff-facing, so use fx node names (a placeholder's name is the user's variable)
    call_expr = f"{name}({', '.join(n.name for n in inputs)})"

    return GeneratedKernel(
        name=name,
        kernel_source=kernel_src,
        wrapper_source=wrapper_src,
        call_expr=call_expr,
    )
