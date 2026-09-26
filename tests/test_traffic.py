"""Tests for analysis/traffic.py - byte accounting for one cluster."""

import torch

from fusion_advisor.analysis.detect_clusters import detect
from fusion_advisor.analysis.traffic import TrafficEstimate, estimate
from fusion_advisor.ir.shapes import propagate
from fusion_advisor.ir.trace import trace
from tests.fixtures import basic


def one_cluster(model, shape):
    gm = trace(model)
    specs = propagate(gm, torch.randn(*shape))
    clusters, _ = detect(gm, specs)
    assert len(clusters) == 1
    return clusters[0], specs


def test_fusion_always_moves_fewer_bytes():
    cluster, specs = one_cluster(basic.ElementwiseChain(), (4, 64))
    est = estimate(cluster, specs)
    assert est.fused_bytes < est.unfused_bytes
    assert 0.0 < est.savings_ratio < 1.0


def test_fused_traffic_is_inputs_plus_outputs():
    """The fused kernel touches memory exactly once per external tensor."""
    cluster, specs = one_cluster(basic.ElementwiseChain(), (4, 64))
    est = estimate(cluster, specs)
    expected = sum(specs[n.name].nbytes for n in cluster.inputs + cluster.outputs)
    assert est.fused_bytes == expected


def test_shared_input_counted_once():
    """`x * 2.0 + x` reads x in two members but the kernel loads it once."""
    cluster, specs = one_cluster(basic.SharedInput(), (4, 64))
    assert [n.name for n in cluster.inputs] == ["x"]
    nbytes = specs["x"].nbytes
    est = estimate(cluster, specs)
    assert est.fused_bytes == 2 * nbytes  # one read of x, one write of add


def test_longer_chain_saves_more():
    """Each fused intermediate removes a write and a read."""
    short, s_specs = one_cluster(basic.RepeatedOperand(), (4, 64))
    long_, l_specs = one_cluster(basic.ElementwiseChain(), (4, 64))
    assert estimate(long_, l_specs).savings_ratio > estimate(short, s_specs).savings_ratio


def test_savings_ratio_handles_empty():
    assert TrafficEstimate(unfused_bytes=0, fused_bytes=0).savings_ratio == 0.0


def test_eval_dropout_moves_no_bytes():
    """gelu -> dropout -> +x: eager runs gelu and add only, 5 tensor passes not 7."""
    cluster, specs = one_cluster(basic.ModuleStyle().eval(), (4, 16, 32))
    t = 4 * 16 * 32 * 4  # every edge is one fp32 [4,16,32] tensor
    est = estimate(cluster, specs)
    assert est.unfused_bytes == 5 * t  # gelu r+w, add r+r+w
    assert est.fused_bytes == 3 * t  # linear and x in, add out


def test_norm_parameters_count_as_traffic():
    """add + layer_norm: 5 tensor passes unfused, 3 fused, plus weight and bias both ways."""
    gm = trace(basic.AddNorm())
    specs = propagate(gm, torch.randn(4, 16, 64), torch.randn(4, 16, 64))
    (cluster,), _ = detect(gm, specs)
    t, p = 4 * 16 * 64 * 4, 64 * 4  # fp32 activation, fp32 [64] parameter
    est = estimate(cluster, specs)
    assert est.unfused_bytes == 5 * t + 2 * p  # add r+r+w, norm r+w + weight + bias
    assert est.fused_bytes == 3 * t + 2 * p  # x, residual, weight, bias in, norm out
