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


@given(graph_sizes(), lists(edges()))
def test_add_edge_sets_edge_count(n_verts, edge_list):
    g = Graph(n_verts)
    for v, w in edge_list:
        assume(v < n_verts and w < n_verts)
        g.add_edge(v, w)
    assert g.E() == len(edge_list)


@given(graph_sizes(), vertices(), vertices())
def test_add_edge_sets_adjacency(n_verts, v, w):
    g = Graph(n_verts)
    assume(v < n_verts and w < n_verts)
    g.add_edge(v, w)
    assert w in g.adj(v)
    assert v in g.adj(w)
