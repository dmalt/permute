"""Statistical utils"""
def compute_cluster_level_stats(stat_map, clusters):
    """
    Compute cluster-level statistic.

    Parameters
    ----------
    stat_map: ndarray
        Statistics computed for each point, i.e. for each time and space
    clusters: list of tuple of ndarray
        Clusters for which cluster-level statisitc is to be computed.
        len of tuple for each cluster should equal stat_map.ndim

    Returns
    -------
    cluster_stats: list of floats
        Cluster-level statistics

    """

    return [stat_map[cluster].sum() for cluster in clusters]
