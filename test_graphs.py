from random import randint
from time import time

from hypothesis import assume, given, settings
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
    seen_edges = []
    for v, w in edge_list:
        assume(v < n_verts and w < n_verts)
        edge = sorted([v, w])
        assume(edge not in seen_edges)
        seen_edges.append(edge)
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


def test_get_components(sample_graph):
    cc = CC(sample_graph)
    comps = cc.get_components()
    assert comps[0] == [0, 1, 2]
    assert comps[1] == [3, 4, 5]


def test_connected_components_on_large_graph():
    n = int(1e6)
    g = Graph(n)
    for _ in range(n):
        v = randint(0, n - 1)
        w = randint(0, n - 1)
        g.add_edge(v, w)
    cc = CC(g)
# --------------------------------------------------------------- #


# ----------- test MaskedSpatioTemporalAdjacencyGraph ----------- #
@fixture(params=[1, 2])
def spatio_temporal_graph_params(request):
    grid_connectivity = lil_matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=bool)
    test_case = request.param
    params = {}
    if test_case == 1:
        params['adj'] = grid_connectivity
        n_spaces = params['adj'].shape[0]
        n_times = 2
        params['mask'] = np.ones((n_times, n_spaces), dtype=bool)
        params['E'] = 10
        params['cc_count'] = 1
    elif test_case == 2:
        params['adj'] = grid_connectivity
        n_spaces = params['adj'].shape[0]
        n_times = 3
        params['mask'] = np.ones((n_times, n_spaces), dtype=bool)
        params['mask'][1, :] = False
        params['E'] = 6
        params['cc_count'] = 2
    yield params


def test_spatio_temporal_graph(spatio_temporal_graph_params):
    """
    Test creation and CC retrieval from masked spatio-temporal graph

    - test the number of vertices in created graph is correct
    - test the number of edges is correct
    - test the number of retrieved connected components is correct

    """
    mask = spatio_temporal_graph_params['mask']
    adjacency = spatio_temporal_graph_params['adj']
    n_times, n_spaces = mask.shape
    stg = MaskedSpatioTemporalAdjacencyGraph(adjacency, mask)
    assert stg.V() == len(mask.nonzero()[0])
    assert stg.E() == spatio_temporal_graph_params['E']
    cc = CC(stg)
    assert cc.count() == spatio_temporal_graph_params['cc_count']
# --------------------------------------------------------------- #
