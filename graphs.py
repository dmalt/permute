"""Graph processing utilities"""
from typing import List, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


class Graph:
    """Adjacency-list graph representation

    Parameters
    ----------
    n_verts : int
        Number of vertices, must be > 0

    """

    def __init__(self, n_verts: int):
        if n_verts <= 0:
            raise ValueError("Number of vertices must be positive integer")
        self._V = n_verts
        self._E = 0
        self._adj = lil_matrix((n_verts, n_verts))

    def __str__(self) -> str:
        res = []
        for i, neighbors in enumerate(self._adj):
            for j in neighbors:
                res.append(f"{i} <--> {j}")
        return "\n".join(res)

    def __repr__(self) -> str:
        return f"Graph(n_verts={self._V})"

    def V(self):
        """
        Get the number of vertices

        """
        return self._V

    def E(self) -> int:
        """
        Get the number of edges

        Returns
        -------
        int
            Number of vertices

        """
        return self._E

    def add_edge(self, v: int, w: int):
        """
        Add edge to the graph

        Parameters
        ----------
        v : int
            First vertex of an edge,  from 0 to n_verts - 1
        w : int
            Second vertex of an edge, from 0 to n_verts - 1

        """
        if not self._adj[v, w] and not self._adj[w, v]:
            self._adj[v, w] = True
            self._adj[w, v] = True
            self._E += 1

    def adj(self, v: int) -> List[int]:
        """
        Get neighbors of vertex v

        Parameters
        ----------
        v : int
            Vertex index, from 0 to n_verts - 1

        Returns
        -------
        list of int
            Neighbors of vertex v

        """
        return self._adj.rows[v]

    def get_adj_matrix(self):
        return self._adj


class CC:
    """
    Find connected components in a graph

    Parameters
    ----------
    graph : Graph
        Graph to process

    """

    def __init__(self, graph: Graph):
        self._count, self._id = connected_components(
            graph.get_adj_matrix(), directed=False, return_labels=True
        )

    def count(self) -> int:
        """
        Get the number of connected components

        Returns
        -------
        int
            Number of connected components
        """
        return self._count

    def id(self, v: int) -> int:
        """
        Given vertex, get index of its connected component

        Parameters
        ----------
        v : int
            Vertex, from 0 to graph.V() - 1

        Returns
        -------
        int
            Index of a connected component
        """
        return self._id[v]

    def get_components(self) -> List[List[int]]:
        """
        Get connected components

        Returns
        -------
        list of int
            Vertices for each connected component

        """
        components = [[] for _ in range(self._count)]
        for v, i_comp in enumerate(self._id):
            components[i_comp].append(v)
        return components


class MaskedSpatioTemporalAdjacencyGraph(Graph):
    """
    Graph defined by selected vertices from spatio-temporal connectivity matrix

    Mask marks vertices above or below certain threshold. Vertices above and
    below threshold are not connected together.

    Parameters
    ----------
    spatial_adjacency : scipy.sparce.spmatrix, shape (n_spaces, n_spaces)
        Adjacency matrix;
    mask : ndarray of -1, 0, 1, shape (n_times, n_spaces)
        Mask of 'active' time-space points

    """

    def __init__(self, spatial_adjacency, mask):
        spatial_adjacency = lil_matrix(spatial_adjacency)
        self._create_maps(mask)
        n_verts = len(mask.nonzero()[0])
        n_times = mask.shape[0]

        super().__init__(n_verts)
        for cur_vert in self._map_lin2mat:
            i_time, i_space = self.lin2mat(cur_vert)
            cur_mask = mask[i_time, i_space]
            # add spatial neighbors
            for v in spatial_adjacency.rows[i_space]:
                if cur_mask != mask[i_time, v]:
                    continue
                neigh_vert = self.mat2lin(i_time, v)
                self.add_edge(cur_vert, neigh_vert)
            # add temporal neighbors
            if i_time > 0 and mask[i_time - 1, i_space] == cur_mask:
                neigh_vert = self.mat2lin(i_time - 1, i_space)
                self.add_edge(cur_vert, neigh_vert)
            if i_time < n_times - 1 and mask[i_time + 1, i_space] == cur_mask:
                neigh_vert = self.mat2lin(i_time + 1, i_space)
                self.add_edge(cur_vert, neigh_vert)

    def lin2mat(self, index: int) -> int:
        """
        Convert linear index to matrix index

        Parameters
        ----------
        index: int
            Linear index

        Returns
        -------
        time_index, space_index: int
            Matrix indices

        """
        return self._map_lin2mat[index]

    def mat2lin(self, i_time: int, i_space: int) -> int:
        """
        Convert matrix index to linear index
        """
        return self._map_mat2lin[(i_time, i_space)]

    def components2mat(
        self, components: List[List[int]]
    ) -> List[List[Tuple[int]]]:
        """
        Convert components` notation from linear to matrix

        Parameters
        ----------
        components: list of list of int
            Connected components in 'linear' notation: each index corresponds
            to a vertex in a processed graph; such vertex indices can't be used
            directly to get time and space indices from the original mask

        Returns
        -------
        list of tuple of ndarray
            Connected components in 'matrix' notation: each vertex is encoded
            by its time and space index.

        """
        converted_components = []
        for component in components:
            times = np.array([self.lin2mat(i)[0] for i in component])
            spaces = np.array([self.lin2mat(i)[1] for i in component])
            converted_components.append((times, spaces))
        return converted_components

    def _create_maps(self, mask):
        self._map_lin2mat = {}
        self._map_mat2lin = {}
        for i, (j, k) in enumerate(zip(*mask.nonzero())):
            self._map_lin2mat[i] = (j, k)
            self._map_mat2lin[(j, k)] = i
