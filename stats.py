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


def compute_cluster_p_value(cluster_stat, perm_max_stats, one_tailed=False):
    """
    Compute p value for a cluter given its distribution for 0-hypothesis

    """
    if one_tailed:
        return sum(ms > cluster_stat for ms in perm_max_stats) / len(
            perm_max_stats
        )
    else:
        return (
            sum(ms > cluster_stat for ms in perm_max_stats)
            + sum(ms < -cluster_stat for ms in perm_max_stats)
        ) / len(perm_max_stats)
def mixed_linear_model(target, regressors, formula, key="confidence"):
    """
    TODO

    Formula example: target ~ is_correct * confidence

    """
    data = regressors.copy()
    data["target"] = target
    md = smf.mixedlm(formula, data=data, groups=regressors[groups_key])
    mdf = md.fit()
    return mdf.tvalues


def find_clusters(data, regressors, stat_fun, adjacency, keys, thresh):
    # -------------------------- compute stats map -------------------------- #
    stats_map = np.empty(data.shape[1:])
    n_times, n_spaces = stats_map.shape
    t_maps = {}
    for i_time, i_space in product(range(n_times), range(n_spaces)):
        t_vals = stat_fun(data[:, i_time, i_space], regressors, formula)
        # compute t_map for each regressor since fitting regression is the most
        # time-consuming part and we get all t-values for free after it
        for key in keys:
            t_maps[key][i_time, i_space] = t_vals[key][i_time, i_space]
    # ----------------------------------------------------------------------- #

    # ---------------------------- find clusters ---------------------------- #
    clusters = {}
    for key in keys:
        mask = t_maps[key] > thresh
        mstag = MaskedSpatioTemporalAdjacencyGraph(adjacency, mask)
        clusters[key] = mstag.components2mat(CC(mstag))
        cluster_stats[key] = compute_cluster_level_stats(
            t_maps[key], clusters[key]
        )
    # ----------------------------------------------------------------------- #
    return t_maps, clusters, cluster_stats


def spatio_temporal_permutation_test_for_correlations(
    data,
    regressors,
    stat_fun,
    adjacency,
    keys,
    thresh,
    group_key="subject",
    n_perm=100,
):
    """
    Perform spatio-temporal permutation test for correlation coefficients.

    Procedure:
    1. Compute t-values for each spatio-temporal point to get stats map
    2. Given the stats map, find clusters
    3. Permute n_perm times. On each permutation:
        a. Compute t-values map for surrogate data
        b. Find clusters
        c. Find maximum cluster-level statistic and store it
    4. For each cluster from step 2 compute p-value with values from step 3a

    Parameters
    ----------
    X: ndarray, shape (n_observations, n_times, n_spaces)
        Data
    regressors: DataFrame
        Regressors
    stat_fun: callable
        Function to compute statistic for each time-space point
    keys: list of str
        Regressor names for which to perform test
    thresh: float
        Threshold for clustering. Only points where stats map is above this
        threshold will be considered during clustering
    n_perm: int
        Number of permutations

    """

    t_maps, clusters, cluster_stats = find_clusters(
        data, regressors, stat_fun, adjacency, keys, thresh
    )
    # --------------------------- do permutations --------------------------- #
    max_cluster_stats = {k: [] for key in keys}
    for _ in trange(n_perm):
        regressors_perm = (
            all_dfs.groupby(group_key).sample(frac=1).reset_index(drop=True)
        )
        _, _, cluster_stats = find_clusters(
            data, regressors, stat_fun, adjacency, keys, thresh
        )
        for key in keys:
            max_cluster_stats[key].append(max(cluster_stats[key]))
    # ----------------------------------------------------------------------- #

    # -------------------- compute p-values for clusters -------------------- #
    pvalues = {key: [] for key in keys}
    for key in keys:
        for cluster_stat in cluster_stats[key]:
            pvalues[key].append(
                compute_cluster_p_value(cluster_stat, max_cluster_stats[key])
            )
    # ----------------------------------------------------------------------- #
