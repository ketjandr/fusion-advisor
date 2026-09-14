"""Tests for cli.py - the full pipeline behind one command."""

import ast
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fusion_advisor import cli
from fusion_advisor.cli import KERNEL_MODULE, main

MODEL = """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def forward(self, x):
        x = x * 2.0
        x = F.relu(x)
        return x + 1.0
"""

TWO_MODELS = MODEL + """
class Other(nn.Module):
    def forward(self, x):
        return x * 3.0
"""

UNTRACEABLE = """
import torch.nn as nn
import torch.nn.functional as F

class Dyn(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return F.relu(x)
        return x
"""

NOTHING_TO_FUSE = """
import torch
import torch.nn as nn

class Solo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 64))
    def forward(self, x):
        return x @ self.w
"""


@pytest.fixture
def runner():
    return CliRunner()


def write(tmp_path, src, name="m.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src))
    return str(p)


def run(runner, tmp_path, src, *args):
    return runner.invoke(main, ["--model", write(tmp_path, src), *args])


def test_happy_path(runner, tmp_path):
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--out-dir", str(tmp_path))
    assert r.exit_code == 0, r.output
    assert "elementwise-chain" in r.output
    assert "Net" in r.output


def test_writes_kernel_file(runner, tmp_path):
    out = tmp_path / "gen"
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--out-dir", str(out))
    assert r.exit_code == 0, r.output
    written = list(out.glob("cluster*.py"))
    assert written, "no kernel file written"
    assert "@triton.jit" in written[0].read_text()


def test_renders_diff(runner, tmp_path):
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--out-dir", str(tmp_path))
    assert "- " in r.output and "+ " in r.output


def test_generation_does_not_prompt(runner, tmp_path):
    out = tmp_path / "gen"
    r = runner.invoke(main, [
        "--model", write(tmp_path, MODEL), "--input-shape", "4,64", "--out-dir", str(out),
    ])
    assert r.exit_code == 0, r.output
    assert "Generate Triton kernels?" not in r.output
    assert list(out.glob("*.py"))


def test_json_payload(runner, tmp_path):
    path = tmp_path / "out.json"
    r = run(
        runner, tmp_path, MODEL,
        "--input-shape", "4,64", "--out-dir", str(tmp_path), "--json", str(path),
    )
    assert r.exit_code == 0, r.output

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    (cluster,) = payload["clusters"]
    assert cluster["category"] == "elementwise-chain"
    assert cluster["traffic"]["unfused_bytes"] > cluster["traffic"]["fused_bytes"]
    assert cluster["mapping"]["quality"] == "exact"
    assert cluster["diff"]["added"]

    # measured only where there is a GPU; the field exists either way
    from fusion_advisor.validate.bench import gpu_available

    if gpu_available():
        assert cluster["validation"]["compiled"] is True
    else:
        assert cluster["validation"] is None


def test_no_fusable_clusters(runner, tmp_path):
    r = run(runner, tmp_path, NOTHING_TO_FUSE, "--input-shape", "4,64")
    assert r.exit_code == 0, r.output
    assert "No fusable clusters" in r.output


def test_explain_rejections_flag(runner, tmp_path):
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64",
            "--out-dir", str(tmp_path), "--explain-rejections")
    assert r.exit_code == 0, r.output
    assert "ejected" in r.output  # header or the "No rejected candidates" line


def test_several_models_is_a_clean_error(runner, tmp_path):
    r = run(runner, tmp_path, TWO_MODELS, "--input-shape", "4,64")
    assert r.exit_code != 0
    assert "--model-class" in r.output
    assert "Traceback" not in r.output


def test_model_class_disambiguates(runner, tmp_path):
    r = run(runner, tmp_path, TWO_MODELS, "--input-shape", "4,64",
            "--out-dir", str(tmp_path), "--model-class", "Other")
    assert r.exit_code == 0, r.output


def test_untraceable_is_a_clean_error(runner, tmp_path):
    r = run(runner, tmp_path, UNTRACEABLE, "--input-shape", "4,64")
    assert r.exit_code != 0
    assert "control flow" in r.output
    assert "Traceback" not in r.output


def test_bad_dtype_is_a_clean_error(runner, tmp_path):
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--dtype", "int9")
    assert r.exit_code != 0
    assert "Unsupported dtype" in r.output


def test_bad_shape_is_a_clean_error(runner, tmp_path):
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,zzz")
    assert r.exit_code != 0
    assert "Bad shape" in r.output


def test_missing_required_options(runner):
    assert runner.invoke(main, []).exit_code != 0


def test_apply_rewrites_the_model(runner, tmp_path):
    path = write(tmp_path, MODEL)
    r = runner.invoke(
        main, ["--model", path, "--input-shape", "4,64", "--apply"], input="y\n"
    )
    assert r.exit_code == 0, r.output

    rewritten = Path(path).read_text()
    assert "cluster0(" in rewritten
    assert "F.relu" not in rewritten  # the fused lines are gone
    assert f"from {KERNEL_MODULE} import cluster0" in rewritten


def test_apply_writes_kernel_module_beside_the_model(runner, tmp_path):
    """The module must be importable from the model, so it lands next to it."""
    path = write(tmp_path, MODEL)
    r = runner.invoke(
        main, ["--model", path, "--input-shape", "4,64", "--apply"], input="y\n"
    )
    assert r.exit_code == 0, r.output
    assert "@triton.jit" in (tmp_path / f"{KERNEL_MODULE}.py").read_text()


def test_apply_leaves_no_backup_files(runner, tmp_path):
    """Edit in place like any modern tool; git is the undo, not a .bak."""
    path = write(tmp_path, MODEL)
    runner.invoke(
        main, ["--model", path, "--input-shape", "4,64", "--apply"], input="y\n"
    )
    assert not list(tmp_path.glob("*.bak"))
    assert not list(tmp_path.glob("*.orig"))


def test_applied_file_is_valid_python(runner, tmp_path):
    path = write(tmp_path, MODEL)
    runner.invoke(
        main, ["--model", path, "--input-shape", "4,64", "--apply"], input="y\n"
    )
    ast.parse(Path(path).read_text())


def test_applied_file_has_no_fx_names(runner, tmp_path):
    """fx names like `linear` are not variables in the user's scope."""
    src = MODEL.replace(
        "    def forward(self, x):\n        x = x * 2.0",
        "    def forward(self, x):\n        x = x * 2.0",
    )
    path = write(tmp_path, src)
    runner.invoke(
        main, ["--model", path, "--input-shape", "4,64", "--apply"], input="y\n"
    )

    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward")
    bound = {a.arg for a in fn.args.args}
    used = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, ast.Store) else used).add(node.id)
    imported = {
        n.name for s in ast.walk(tree) if isinstance(s, ast.ImportFrom) for n in s.names
    }
    assert not (used - bound - imported), "forward() references an undefined name"


def test_apply_declining_prompt_leaves_file_alone(runner, tmp_path):
    path = write(tmp_path, MODEL)
    original = Path(path).read_text()
    r = runner.invoke(main, ["--model", path, "--input-shape", "4,64", "--apply"], input="n\n")
    assert r.exit_code == 0, r.output
    assert Path(path).read_text() == original
    assert not (tmp_path / f"{KERNEL_MODULE}.py").exists()


def test_apply_without_clusters_changes_nothing(runner, tmp_path):
    path = write(tmp_path, NOTHING_TO_FUSE)
    original = Path(path).read_text()
    r = runner.invoke(main, ["--model", path, "--input-shape", "4,64", "--apply"])
    assert r.exit_code == 0, r.output
    assert Path(path).read_text() == original


def test_says_why_it_did_not_measure(runner, tmp_path):
    """Without a GPU the run must say so, not silently omit the benchmark."""
    from fusion_advisor.validate.bench import gpu_available

    if gpu_available():
        pytest.skip("this asserts the no-GPU path")
    r = run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--out-dir", str(tmp_path))
    assert "Not measured" in r.output


def test_vs_inductor_reaches_validate(runner, tmp_path, monkeypatch):
    """The flag is opt-in because each torch.compile costs seconds."""
    seen = {}

    def fake_validate(kernel, cluster, gm, specs, **kw):
        seen.update(kw)
        return SimpleNamespace(skipped_reason="stubbed", usable=False, benchmark=None)

    monkeypatch.setattr(cli, "validate", fake_validate)
    monkeypatch.setattr(cli, "gpu_available", lambda: True)
    run(runner, tmp_path, MODEL, "--input-shape", "4,64",
        "--out-dir", str(tmp_path), "--vs-inductor")
    assert seen["vs_inductor"] is True


def test_vs_inductor_defaults_off(runner, tmp_path, monkeypatch):
    seen = {}

    def fake_validate(kernel, cluster, gm, specs, **kw):
        seen.update(kw)
        return SimpleNamespace(skipped_reason="stubbed", usable=False, benchmark=None)

    monkeypatch.setattr(cli, "validate", fake_validate)
    monkeypatch.setattr(cli, "gpu_available", lambda: True)
    run(runner, tmp_path, MODEL, "--input-shape", "4,64", "--out-dir", str(tmp_path))
    assert seen["vs_inductor"] is False
