"""Tests for loader.py - importing a user's file and resolving its model."""

import textwrap
from pathlib import Path

import pytest
import torch

from fusion_advisor.loader import (
    LoadError,
    build_example_input,
    load,
    parse_dtype,
    parse_shape,
)

ONE_MODEL = """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def forward(self, x):
        return F.relu(x) + 1.0
"""

TWO_MODELS = ONE_MODEL + """
class Other(nn.Module):
    def forward(self, x):
        return x * 2.0
"""

IMPORTS_CLASSES = """
from torch.nn import GELU, Linear, Module

class Mine(Module):
    def __init__(self):
        super().__init__()
        self.fc, self.act = Linear(8, 8), GELU()
    def forward(self, x):
        return self.act(self.fc(x))
"""

NEEDS_ARGS = """
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w = width
    def forward(self, x):
        return x
"""


def write(tmp_path, src, name="model.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src))
    return str(p)


def test_single_model_resolves(tmp_path):
    lm = load(write(tmp_path, ONE_MODEL))
    assert type(lm.module).__name__ == "Net"
    assert isinstance(lm.module, torch.nn.Module)


def test_model_is_eval_mode(tmp_path):
    """Inference-only, so dropout is identity like the lowering rule assumes."""
    assert load(write(tmp_path, ONE_MODEL)).module.training is False


def test_source_text_matches_file(tmp_path):
    path = write(tmp_path, ONE_MODEL)
    lm = load(path)
    assert lm.source_text == Path(path).read_text()
    assert lm.source_path == path


def test_imported_classes_are_not_candidates(tmp_path):
    """`from torch.nn import Linear` must not make Linear a candidate."""
    assert type(load(write(tmp_path, IMPORTS_CLASSES)).module).__name__ == "Mine"


def test_several_models_refuses_to_guess(tmp_path):
    with pytest.raises(LoadError, match="pass --model-class"):
        load(write(tmp_path, TWO_MODELS))


def test_class_name_disambiguates(tmp_path):
    path = write(tmp_path, TWO_MODELS)
    assert type(load(path, "Other").module).__name__ == "Other"


def test_unknown_class_name_lists_options(tmp_path):
    with pytest.raises(LoadError, match="Net"):
        load(write(tmp_path, TWO_MODELS), "Missing")


def test_no_model_in_file(tmp_path):
    with pytest.raises(LoadError, match="No nn.Module subclass"):
        load(write(tmp_path, "x = 1\n"))


def test_missing_file():
    with pytest.raises(LoadError, match="No such file"):
        load("definitely/not/here.py")


def test_import_error_is_wrapped(tmp_path):
    with pytest.raises(LoadError, match="Importing"):
        load(write(tmp_path, "raise RuntimeError('boom')\n"))


def test_constructor_args_reported(tmp_path):
    with pytest.raises(LoadError, match="constructor arguments"):
        load(write(tmp_path, NEEDS_ARGS))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("4,128", (4, 128)), ("4, 128", (4, 128)), ("8", (8,)), ("2,3,4,5", (2, 3, 4, 5))],
)
def test_parse_shape(text, expected):
    assert parse_shape(text) == expected


@pytest.mark.parametrize("bad", ["", "4,0", "4,-1", "4,x"])
def test_parse_shape_rejects(bad):
    with pytest.raises(LoadError):
        parse_shape(bad)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("float32", torch.float32), ("fp16", torch.float16), ("torch.bfloat16", torch.bfloat16)],
)
def test_parse_dtype(text, expected):
    assert parse_dtype(text) is expected


def test_parse_dtype_rejects():
    with pytest.raises(LoadError, match="Unsupported dtype"):
        parse_dtype("int9")


def test_build_example_input():
    t = build_example_input((2, 8), torch.float16)
    assert t.shape == (2, 8)
    assert t.dtype is torch.float16
