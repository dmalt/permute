from hypothesis import given, settings, assume
from hypothesis.strategies import integers, lists, tuples
from pytest import fixture, raises

from graphs import Graph, CC


# ------------------------- test Graph  ------------------------- #
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

# --------------------------------------------------------------- #


# --------------------------- test CC --------------------------- #
@fixture
def sample_graph():
    """Create graph with two connected components"""
    g = Graph(6)

    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    g.add_edge(3, 4)
    g.add_edge(4, 5)
    return g

def test_number_of_components_correct(sample_graph):
    cc = CC(sample_graph)
    assert cc.count() == 2

def test_vertices_get_correct_cc_ids(sample_graph):
    cc = CC(sample_graph)

    assert cc.id(0) == 0
    assert cc.id(1) == 0
    assert cc.id(2) == 0
    assert cc.id(3) == 1
    assert cc.id(4) == 1
    assert cc.id(5) == 1

# --------------------------------------------------------------- #
