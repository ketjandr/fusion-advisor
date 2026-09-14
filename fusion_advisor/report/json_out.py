"""JSON output - the machine-readable contract for editors and CI, so version it."""

from __future__ import annotations

SCHEMA_VERSION = 1


def _diff_payload(diff) -> dict | None:
    if diff is None:
        return None
    return {
        "file": diff.file,
        "start_line": diff.start_line,
        "end_line": diff.end_line,
        "removed": diff.removed,
        "added": diff.added,
    }


def _validation_payload(v) -> dict | None:
    """Flatten ValidationResult; None until a GPU run fills it in."""
    if v is None:
        return None
    b = v.benchmark
    return {
        "compiled": v.compiled,
        "numerics_ok": v.numerics_ok,
        "model_numerics_ok": v.model_numerics_ok,
        "usable": v.usable,
        "max_abs_err": v.max_abs_err,
        "skipped_reason": v.skipped_reason,
        "benchmark": None
        if b is None
        else {
            # regime qualifies the speedup; launch-bound numbers are not bandwidth
            "regime": b.regime.value,
            "working_set_bytes": b.working_set_bytes,
            "cluster_speedup": round(b.cluster_speedup, 4),
            "model_speedup": None if b.model_speedup is None else round(b.model_speedup, 4),
            "achieved_gbps": round(b.cluster_fused.achieved_gbps, 1),
            "peak_gbps": b.peak_gbps,
            "peak_source": b.peak_source,
            "pct_of_peak": None if b.pct_of_peak is None else round(b.pct_of_peak, 1),
            "vs_inductor": None if b.vs_inductor is None else round(b.vs_inductor, 4),
        },
    }


def _cluster_payload(cluster, est, rng, diff, validation) -> dict:
    return {
        "index": cluster.index,
        "category": cluster.category.value,
        "nodes": [n.name for n in cluster.nodes],
        "inputs": [n.name for n in cluster.inputs],
        "outputs": [n.name for n in cluster.outputs],
        "traffic": {
            "unfused_bytes": est.unfused_bytes,
            "fused_bytes": est.fused_bytes,
            "savings_ratio": round(est.savings_ratio, 4),
            "basis": "dram-upper-bound",  # no cache model, so not a speedup claim
        },
        "mapping": {
            "quality": rng.quality.value,
            "file": rng.file,
            "start_line": rng.start_line,
            "end_line": rng.end_line,
        },
        "diff": _diff_payload(diff),
        "validation": _validation_payload(validation),
    }


def build_payload(clusters, estimates, ranges, diffs, validations, rejections) -> dict:
    """Per cluster: category, nodes, byte estimates, mapping quality, diff, validation."""
    return {
        "schema_version": SCHEMA_VERSION,
        "clusters": [
            _cluster_payload(c, e, r, d, v)
            for c, e, r, d, v in zip(clusters, estimates, ranges, diffs, validations, strict=True)
        ],
        "rejections": [
            {"nodes": [n.name for n in r.nodes], "reason": r.reason.value, "detail": r.detail}
            for r in rejections
        ],
    }
