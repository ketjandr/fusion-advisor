import pytest

xfail = pytest.mark.xfail(reason="scaffolding", strict=False)


@xfail
def test_flat_functional_model_maps_exactly():
    raise NotImplementedError


@xfail
def test_deepest_user_frame_wins():
    """Nested modules: the first frame is the caller, the last is the author."""
    raise NotImplementedError


@xfail
def test_node_inside_torch_is_unavailable():
    """No user frame means no diff to show."""
    raise NotImplementedError


@xfail
def test_cluster_spanning_two_functions_is_approximate():
    raise NotImplementedError


@xfail
def test_chained_one_liner_does_not_render_diff():
    """A shared line holds non-cluster nodes, so replacing it deletes them."""
    raise NotImplementedError


@xfail
def test_real_transformer_block_mapping_quality():
    """What fraction of clusters actually map EXACT?"""
    raise NotImplementedError
