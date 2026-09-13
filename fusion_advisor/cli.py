"""click entry point.

load -> trace -> shape-prop -> detect -> estimate -> report
                                     -> [y/n] emit -> validate -> diff
"""

from __future__ import annotations

import click


@click.command()
@click.option("--model", required=True, help="Path to a .py file defining an nn.Module.")
@click.option("--model-class", default=None, help="Class name, if the file defines several.")
@click.option("--input-shape", required=True, help="Comma-separated example input shape, e.g. 4,128")
@click.option("--dtype", default="float32")
@click.option("--json", "json_out", default=None, help="Write machine-readable results here.")
@click.option("--explain-rejections", is_flag=True, help="Show why candidates were not fused.")
@click.option("--yes", is_flag=True, help="Generate kernels without prompting.")
@click.option(
    "--vs-inductor",
    is_flag=True,
    help="Also benchmark torch.compile per cluster. Slow.",
)
@click.option("--out-dir", default=".", help="Where to write generated kernel files.")
def main(
    model, model_class, input_shape, dtype, json_out, explain_rejections, yes, vs_inductor, out_dir
):
    """Fusion Advisor - epilogue-fusion analysis for PyTorch models."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
