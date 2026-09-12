import pytest

xfail = pytest.mark.xfail(reason="scaffolding", strict=False)


@xfail
def test_elementwise_chain_forms_one_cluster():
    raise NotImplementedError


@xfail
def test_reconverging_diamond_is_one_cluster():
    """Fork + rejoin is fusable; len(users) > 1 wrongly rejects it."""
    raise NotImplementedError


@xfail
def test_opaque_consumer_blocks_fusion():
    """TrueFanOut: relu also feeds a matmul, so it must be materialized."""
    raise NotImplementedError


@xfail
def test_escaping_output_blocks_fusion():
    raise NotImplementedError


@xfail
def test_repeated_operand_is_legal():
    """`h + h` - one user, two arg positions."""
    raise NotImplementedError


@xfail
def test_non_convex_cluster_rejected():
    """relu → matmul → add: fusing {relu, mul, add} is unschedulable."""
    raise NotImplementedError


@xfail
def test_matmul_splits_into_two_clusters():
    raise NotImplementedError


@xfail
def test_unknown_op_defaults_to_opaque():
    """Unrecognized targets must never be absorbed."""
    raise NotImplementedError
