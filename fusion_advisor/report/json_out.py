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
        "validation": validation,  # None until a GPU run fills it in
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
