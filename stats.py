"""Statistical utils"""
from itertools import product

import numpy as np
from tqdm import trange, tqdm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from graphs import CC, MaskedSpatioTemporalAdjacencyGraph


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


def compute_cluster_p_value(cluster_stat, cluster_stats_H0, tail):
    """
    Compute p value for a cluter given its distribution for 0-hypothesis

    """
    if tail == 1:
        return sum(ms > cluster_stat for ms in cluster_stats_H0) / len(
            cluster_stats_H0
        )
    elif tail == -1:
        return sum(ms < cluster_stat for ms in cluster_stats_H0) / len(
            cluster_stats_H0
        )
    else:
        return sum(abs(ms) > cluster_stat for ms in cluster_stats_H0) / len(
            cluster_stats_H0
        )


def mixed_linear_model(target, regressors, formula, groups_key):
    """
    TODO

    Formula example: target ~ is_correct * confidence

    """
    data = regressors.copy()
    data["target"] = target
    md = smf.mixedlm(formula, data=data, groups=regressors[groups_key])
    mdf = md.fit()
    return mdf.tvalues


def find_clusters(data, regressors, stat_fun, adjacency, keys, thresh, tail):
    # -------------------------- compute stats map -------------------------- #
    stats_map = np.empty(data.shape[1:])
    n_times, n_spaces = stats_map.shape
    t_maps = {key: np.empty((n_times, n_spaces)) for key in keys}
    # for i_time, i_space in tqdm(product(range(n_times), range(n_spaces))):
    #     t_vals = stat_fun(data[:, i_time, i_space], regressors)
    #     # compute t_map for each regressor since fitting regression is the most
    #     # time-consuming part and we get all t-values for free after it
    #     for key in keys:
    #         t_maps[key][i_time, i_space] = t_vals[key]
    t_vals = Parallel(n_jobs=-1)(
        delayed(stat_fun)(data[:, i_time, i_space], regressors)
        for i_time, i_space in tqdm(
            product(range(n_times), range(n_spaces)), total=n_times * n_spaces
        )
    )

    for k, (i_time, i_space) in tqdm(
        enumerate(product(range(n_times), range(n_spaces)))
    ):
        for key in keys:
            t_maps[key][i_time, i_space] = t_vals[k][key]
    # ----------------------------------------------------------------------- #

    # ---------------------------- find clusters ---------------------------- #
    clusters = {}
    cluster_stats = {}
    for key in keys:
        if tail == 1:
            mask = (t_maps[key] > thresh).astype(int)
        elif tail == -1:
            mask = (t_maps[key] < thresh).astype(int)
        else:
            mask = (t_maps[key] > thresh).astype(int) - (
                t_maps[key] < -thresh
            ).astype(int)
        if np.count_nonzero(mask):
            mstag = MaskedSpatioTemporalAdjacencyGraph(adjacency, mask)
            clusters[key] = mstag.components2mat(CC(mstag).get_components())
            cluster_stats[key] = compute_cluster_level_stats(
                t_maps[key], clusters[key]
            )
        else:
            clusters[key] = []
            cluster_stats[key] = []
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
    tail=0,
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

    t_maps, clusters, cluster_stats_orig = find_clusters(
        data, regressors, stat_fun, adjacency, keys, thresh, tail
    )
    # --------------------------- do permutations --------------------------- #
    cluster_stats_H0 = {key: [] for key in keys}
    for _ in trange(n_perm):
        regressors_perm = (
            regressors.groupby(group_key).sample(frac=1).reset_index(drop=True)
        )
        _, _, cluster_stats = find_clusters(
            data, regressors_perm, stat_fun, adjacency, keys, thresh, tail
        )
        for key in keys:
            if cluster_stats[key]:
                if tail == 1:
                    cluster_stats_H0[key].append(max(cluster_stats[key]))
                elif tail == -1:
                    cluster_stats_H0[key].append(min(cluster_stats[key]))
                else:
                    cluster_stats_H0[key].append(
                        max(map(abs, cluster_stats[key]))
                    )
            else:
                cluster_stats_H0[key].append(0)
    # ----------------------------------------------------------------------- #

    # -------------------- compute p-values for clusters -------------------- #
    pvalues = {key: [] for key in keys}
    for key in keys:
        for cluster_stat_orig in cluster_stats_orig[key]:
            pvalues[key].append(
                compute_cluster_p_value(
                    cluster_stat_orig, cluster_stats_H0[key], tail
                )
            )
    # ----------------------------------------------------------------------- #
    # assert False
    return t_maps, clusters, pvalues, cluster_stats_H0


if __name__ == "__main__":
    import pandas as pd
    from mne.io import read_info
    from mne.channels import find_ch_adjacency
    from functools import partial

    print("Reading data ...")
    data = np.load("data.npy")
    print("Done")
    data_short = np.sqrt(
        data[:, 100:103, 0::2] ** 2 + data[:, 100:103, 1::2] ** 2
    )
    all_dfs = pd.read_pickle("regressors.pkl")
    info = read_info("info.fif")
    adjacency, ch_names = find_ch_adjacency(info, ch_type="mag")
    stat_fun = partial(
        mixed_linear_model,
        formula="target ~ confidence * is_correct",
        groups_key="subject",
    )
    res = spatio_temporal_permutation_test_for_correlations(
        data_short,
        all_dfs,
        stat_fun,
        adjacency,
        ["confidence"],
        2,
        "subject",
        n_perm=10,
    )
