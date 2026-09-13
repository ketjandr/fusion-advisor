from __future__ import annotations

from dataclasses import dataclass

import torch.fx as fx

from fusion_advisor.ir.op_registry import OpCategory, classify, reduction_axis

from ..analysis.cluster import ClusterCategory, FusableCluster
from ..analysis.legality import escaping_nodes
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


def emit(cluster: FusableCluster, specs: dict) -> GeneratedKernel:
    """Emit a Triton kernel + wrapper for one cluster."""
    nodes = cluster.nodes
    cluster_set = set(nodes)
    escaping = escaping_nodes(nodes)
    name = f"cluster{cluster.index}"

    # assign variable names
    node_to_var: dict[fx.Node, str] = {}
    var_count = 0

    # external inputs: cluster node args that come from outside
    external_inputs: list[fx.Node] = []
    seen_inputs: set[fx.Node] = set()
    for n in nodes:
        for a in n.args:
            if isinstance(a, fx.Node) and a not in cluster_set and a not in seen_inputs:
                seen_inputs.add(a)
                external_inputs.append(a)

    # tl.load lines (one per external input)
    load_lines: list[str] = []
    in_ptrs: list[str] = []
    for i, inp in enumerate(external_inputs):
        ptr = f"in_ptr{i}"
        var = f"v{var_count}"
        in_ptrs.append(ptr)
        node_to_var[inp] = var
        load_lines.append(f"{var} = tl.load({ptr} + offs, mask=mask)")
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
    wrapper_lines = [
        f"def {name}({', '.join(in_args)}):",
        f"    out0 = torch.empty_like({in_args[0]})",
        f"    n = {in_args[0]}.numel()",
        f"    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)",
        f"    {name}_kernel[grid]({', '.join(in_args)}, out0, n, BLOCK_SIZE=1024)",
        f"    return out0",
    ]
    wrapper_src = "\n".join(wrapper_lines)

    # diff-facing, so use fx node names (a placeholder's name is the user's variable)
    call_expr = f"{name}({', '.join(n.name for n in external_inputs)})"

    return GeneratedKernel(
        name=name,
        kernel_source=kernel_src,
        wrapper_source=wrapper_src,
        call_expr=call_expr,
    )
