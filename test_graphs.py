from hypothesis import given, settings
from hypothesis.strategies import integers
from pytest import raises

from graphs import Graph


@settings(deadline=500)
@given(integers(min_value=1, max_value=5e5))
def test_normal_graph_creation(n_verts):
    g = Graph(n_verts)
    assert g.V() == n_verts


@settings(deadline=500)
@given(integers(min_value=-5e5, max_value=0))
def test_creation_raises_value_error_for_negative_input(n_verts):
    with raises(ValueError):
        g = Graph(n_verts)

