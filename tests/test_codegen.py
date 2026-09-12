import pytest

xfail = pytest.mark.xfail(reason="scaffolding", strict=False)


@xfail
def test_topological_substitution_chains_variables():
    raise NotImplementedError


@xfail
def test_emitted_kernel_is_valid_python_syntax():
    """ast.parse catches most emitter bugs without a GPU."""
    raise NotImplementedError


@xfail
def test_grid_sizing_derives_from_shapes_not_constants():
    raise NotImplementedError


@xfail
def test_broadcast_operand_gets_its_own_index():
    """[D] bias against [B,S,D] can't share the flat offset."""
    raise NotImplementedError


@xfail
def test_missing_lowering_rule_is_rejected_not_crashed():
    raise NotImplementedError
