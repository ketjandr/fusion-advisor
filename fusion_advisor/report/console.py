"""Terse rich console output - one line per cluster, no prose."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..sourcemap.provenance import MappingQuality
from ..validate.bench import PEAK_GBPS, CacheRegime

console = Console(highlight=False)  # auto-styling splits "2.00x" across colour codes

_QUALITY_STYLE = {
    MappingQuality.EXACT: "green",
    MappingQuality.APPROXIMATE: "yellow",
    MappingQuality.UNAVAILABLE: "dim",
}


def human_bytes(n: int) -> str:
    """Byte count as B/KB/MB/GB."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _where(rng) -> str:
    """Line range if we mapped it, else the op chain."""
    if rng is None or rng.start_line is None:
        return rng.fallback_label if rng else "-"
    if rng.start_line == rng.end_line:
        return f"L{rng.start_line}"
    return f"L{rng.start_line}-{rng.end_line}"


def render_clusters(clusters, estimates, ranges=None) -> None:
    """Per-cluster traffic table; totals are absolute bytes, never an averaged percent."""
    if not clusters:
        console.print("[yellow]No fusable clusters found.[/yellow]")
        return

    ranges = ranges or [None] * len(clusters)
    table = Table(
        title="Fusable clusters (estimated memory traffic)",
        title_justify="left",
        header_style="",
    )
    columns = (
        ("#\n", "left"),
        ("category\n", "left"),
        ("ops\n", "left"),
        ("traffic\n(unfused)", "right"),
        ("traffic\n(fused)", "right"),
        ("traffic\nsaved", "right"),
        ("map\n", "left"),
        ("lines\n", "left"),
    )
    for name, justify in columns:
        table.add_column(name, justify=justify)

    for c, est, rng in zip(clusters, estimates, ranges, strict=True):
        quality = rng.quality if rng else None
        table.add_row(
            str(c.index),
            c.category.value,
            str(len(c.nodes)),
            human_bytes(est.unfused_bytes),
            human_bytes(est.fused_bytes),
            f"{est.savings_ratio:.0%}",
            f"[{_QUALITY_STYLE[quality]}]{quality.value}[/]" if quality else "-",
            _where(rng),
        )
    console.print(table)

    saved = sum(e.unfused_bytes - e.fused_bytes for e in estimates)
    total = sum(e.unfused_bytes for e in estimates)
    console.print(
        f"Across {len(clusters)} cluster(s): {human_bytes(saved)} of {human_bytes(total)} "
        f"cluster-local traffic avoided.",
        style="dim",
    )
    console.print("Estimates are a DRAM upper bound with no cache model.", style="dim italic")


def render_rejections(rejections) -> None:
    """--explain-rejections: why candidates were refused."""
    if not rejections:
        console.print("No rejected candidates.", style="dim")
        return

    table = Table(title="Rejected candidates", title_justify="left", header_style="bold")
    table.add_column("ops")
    table.add_column("reason", style="yellow")
    for r in rejections:
        table.add_row(", ".join(n.name for n in r.nodes), r.reason.value)
    console.print(table)


def render_diff(cluster_diff) -> None:
    """Red/green replacement for one cluster."""
    if cluster_diff is None:
        return
    d = cluster_diff
    console.print(f"\n[bold]{d.file}[/bold]:{d.start_line}-{d.end_line}")
    for line in d.removed:
        console.print(f"[red]- {line}[/red]")
    for line in d.added:
        console.print(f"[green]+ {line}[/green]")


def render_validation(cluster, result) -> None:
    """Measured numerics and speedup for one cluster, or why there are none."""
    if result.skipped_reason:
        console.print(f"cluster {cluster.index}: not measured - {result.skipped_reason}", style="dim")
        return
    if not result.usable:
        console.print(
            f"cluster {cluster.index}: NUMERICS FAILED (max abs err {result.max_abs_err:.2e}) "
            f"- do not use this kernel",
            style="bold red",
        )
        return

    b = result.benchmark
    console.print(
        f"cluster {cluster.index}: verified (max abs err {result.max_abs_err:.2e})", style="green"
    )
    if b.regime is CacheRegime.LAUNCH_BOUND:
        # a ratio here is overhead noise; printing it as a speedup would mislead
        console.print(
            f"  {b.regime.value} at {human_bytes(b.working_set_bytes)} - too small to measure "
            f"fusion. Re-run with a larger --input-shape.",
            style="yellow",
        )
        return

    console.print(
        f"  {b.regime.value}  {human_bytes(b.working_set_bytes)}  "
        f"fused cluster speedup [bold]{b.cluster_speedup:.2f}x[/bold]  "
        f"model speedup [bold]{b.model_speedup:.2f}x[/bold]  "
        f"fused cluster bandwidth {b.cluster_fused.achieved_gbps:.0f} GB/s "
        f"({b.cluster_fused.pct_of_peak:.0f}% of {PEAK_GBPS:.0f} GB/s peak memory bandwidth)"
    )
    if b.cluster_inductor is not None:
        # >1 means our kernel beat what torch.compile produced
        console.print(f"  vs torch.compile: [bold]{b.vs_inductor:.2f}x[/bold]", style="dim")


def render_unmapped(cluster, rng) -> None:
    """Why a cluster got no diff, so a missing diff never looks like a crash."""
    reason = (
        "no user source frame"
        if rng.quality is MappingQuality.UNAVAILABLE
        else "line range holds code outside the cluster, or spans two functions"
    )
    console.print(f"\ncluster {cluster.index} ({_where(rng)}): no safe diff - {reason}", style="dim")
