"""Subgraph extraction and in-memory model rewrite, for measurement only."""

from __future__ import annotations

import copy
import operator

import torch
import torch.fx as fx


def extract_subgraph(gm: fx.GraphModule, cluster) -> fx.GraphModule:
    """Standalone GraphModule computing exactly this cluster.

    Deriving the eager baseline from the real graph, rather than hand-writing a
    reference per fixture, stops every speedup being measured against a strawman.
    """
    graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}

    # every external value becomes a parameter, including get_attr weights
    for inp in cluster.inputs:
        env[inp] = graph.placeholder(inp.name)
    for node in cluster.nodes:
        env[node] = graph.node_copy(node, lambda n: env[n])

    outs = [env[o] for o in cluster.outputs]
    graph.output(outs[0] if len(outs) == 1 else tuple(outs))
    graph.lint()
    return fx.GraphModule(gm, graph)


def rewrite_with_kernel(gm: fx.GraphModule, cluster, wrapper) -> fx.GraphModule:
    """Copy of `gm` with the cluster replaced by a call to `wrapper`."""
    patched = copy.deepcopy(gm)
    # deepcopy gives new Node objects, so re-find them by name
    by_name = {n.name: n for n in patched.graph.nodes}

    members = [by_name[n.name] for n in cluster.nodes]
    outs = [by_name[o.name] for o in cluster.outputs]
    args = tuple(by_name[n.name] for n in cluster.inputs)

    with patched.graph.inserting_after(members[-1]):  # after every member, so inputs exist
        call = patched.graph.call_function(wrapper, args)
    if len(outs) == 1:
        outs[0].replace_all_uses_with(call)
    else:  # the kernel returns a tuple; each output feeds its own consumers
        for i, out in reversed(list(enumerate(outs))):
            with patched.graph.inserting_after(call):
                item = patched.graph.call_function(operator.getitem, (call, i))
            out.replace_all_uses_with(item)

    # reverse topological order, so no node is erased while still used
    for node in reversed(members):
        patched.graph.erase_node(node)

    patched.graph.lint()
    patched.recompile()
    return patched


def allocate_inputs(cluster, specs, device="cuda") -> list[torch.Tensor]:
    """Fresh inputs matching the cluster's external TensorSpecs.

    Use the real dtype, not a float32 default - it changes both bytes moved and
    whether reductions need fp32 accumulation.
    """
    out = []
    for node in cluster.inputs:
        spec = specs[node.name]
        if spec.dtype.is_floating_point:
            out.append(torch.randn(spec.dims, dtype=spec.dtype, device=device))
        elif spec.dtype is torch.bool:
            out.append(torch.randint(0, 2, spec.dims, device=device).bool())
        else:
            out.append(torch.ones(spec.dims, dtype=spec.dtype, device=device))
    return out


def working_set_bytes(cluster, specs) -> int:
    """Bytes the fused kernel actually touches: external reads plus stored writes."""
    return sum(
        specs[n.name].nbytes for n in cluster.inputs + cluster.outputs if n.name in specs
    )
