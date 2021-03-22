from typing import List

from scipy.sparse import lil_matrix, csr_matrix


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
        return csr_matrix(self._adj)


class CC:
    """
    Find connected components in a graph

    Parameters
    ----------
    graph : Graph
        Graph to process

    """

    def __init__(self, graph: Graph):
        self._marked = [None] * graph.V()
        self._id = [None] * graph.V()
        self._count = 0
        for v in range(graph.V()):
            if not self._marked[v]:
                self._dfs(graph, v)
                self._count += 1

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
        components = [[] for _ in self._count]
        for v, i_comp in enumerate(self._id):
            components[i_comp].append(v)
        return components

    def _dfs(self, graph: Graph, v: int):
        """Mark connected components with Depth-First Search"""
        self._marked[v] = True
        self._id[v] = self._count
        for w in graph.adj(v):
            if not self._marked[w]:
                self._dfs(graph, w)
