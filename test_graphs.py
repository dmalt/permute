from hypothesis import given, settings, assume
from hypothesis.strategies import integers, lists, tuples
from pytest import raises

from graphs import Graph


MAX_VERTS = 1e5

def graph_sizes():
    return integers(min_value=1, max_value=MAX_VERTS)

def vertices():
    return integers(min_value=0, max_value=MAX_VERTS - 1)

def edges():
    return tuples(vertices(), vertices())


@given(graph_sizes())
def test_normal_graph_creation(n_verts):
    g = Graph(n_verts)
    assert g.V() == n_verts


@given(integers(min_value=-MAX_VERTS, max_value=0))
def test_creation_raises_value_error_for_negative_input(n_verts):
    with raises(ValueError):
        g = Graph(n_verts)

