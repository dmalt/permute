class Graph:
    """Adjacency-list graph representation

    Parameters
    ----------
    n_verts : int
        Number of vertices

    """
    def __init__(self, n_verts: int):
        if n_verts <= 0:
            raise ValueError("Number of vertices should be positive integer")
        self._V = n_verts
        self._E = 0
        self._adj = [[] for _ in range(n_verts)]

    def __str__(self):
        res = []
        for i, neighbors in enumerate(self._adj):
            for j in neighbors:
                res.append(f"{i} <--> {j}")
        return "\n".join(res)

    def __repr__(self):
        return f"Graph(n_verts={self._V})"

    def V(self):
        """
        Return the number of vertices
        """
        return self._V

    def E(self):
        """
        Return the number of edges
        """
        return self._E

    def add_edge(self, v: int, w: int):
        """
        Add edge to the graph
        """
        self._adj[v].append(w)
        self._adj[w].append(v)
        self._E += 1


if __name__ == "__main__":
    g = Graph(4)
    print(g)
