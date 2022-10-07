#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:06:09 2020

@author: vito
"""

import numpy as np

from sklearn.cluster import DBSCAN
from hdbscan.hdbscan_ import HDBSCAN
from ray import tune

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb  # noqa
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import math

import numpy as np
import os

from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.RI import rand_score
from sklearn.model_selection import train_test_split

def Tuning_Hyperparametes(X, Y, dataset_name):

    shrink = 0.01
    _, X, _, Y = train_test_split(X, Y, test_size = shrink)

    log_folder = "./result/" + dataset_name + "/config_performance.txt"
    directory = os.path.dirname(log_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    def train_TGMRF(config):
        # if config["width"] < config["stride"]:
        #     return
        clf = TGMRF(epsilon=55, width=32, stride=8, maxIters=30, lr=0, lamb=config["lamb"], beta=config["beta"])
        icspca, icspca_row, last, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = clf.fit_transform(X)
        
        
        _eps = np.percentile(icspca.reshape(-1)[icspca.reshape(-1) != 0], 6)
        clustering_dbscan = DBSCAN(eps=_eps, min_samples=3, metric="precomputed").fit(icspca)
        clustering_hdbscan = HDBSCAN(min_cluster_size=3, metric='precomputed').fit_predict(icspca)

        clustering_MD = MD_Cluster(diff_threshold=0.0015, slope_threshold=0.015)
        clustering_result_md = clustering_MD.fit_predict(icspca)

        if icspca_row.shape[1] > 30:
            pca = PCA(n_components=min(20, min(icspca_row.shape[0], icspca_row.shape[1])))
            icspca_row = pca.fit_transform(icspca_row)

        ri_dbscan = rand_score(clustering_dbscan.labels_, Y)
        ri_hdbscan = rand_score(clustering_hdbscan, Y)
        ri_md = rand_score(clustering_result_md, Y)
        ari_dbscan = adjusted_rand_score(clustering_dbscan.labels_, Y)
        ari_hdbscan = adjusted_rand_score(clustering_hdbscan, Y)
        ari_md = adjusted_rand_score(clustering_result_md, Y)
        # csm_dbscan = cluster_similarity_measure(clustering_dbscan.labels_, Y, "dataset")
        # csm_hdbscan  = cluster_similarity_measure(clustering_hdbscan, Y, "dataset")
        nmi_dbscan = normalized_mutual_info_score(clustering_dbscan.labels_, Y, average_method="max")
        nmi_hdbscan = normalized_mutual_info_score(clustering_hdbscan, Y, average_method="max")
        nmi_md = normalized_mutual_info_score(clustering_result_md, Y, average_method="max")

        shrink = 1.0
        bic = math.log(X.shape[0]) * numberOfParameters * shrink - 2 * math.log(aggregated_ll_Loss + aggregated_penalty_loss)

        if len(clustering_hdbscan) == 1:
            clustering_hdbscan[0] = 20
        silhouette_hdbscan = silhouette_score(icspca, clustering_hdbscan, metric="precomputed")
        if len(clustering_MD) == 1:
            clustering_MD[0] = 20
        silhouette_md = silhouette_score(icspca, clustering_result_md, metric="precomputed")


        tune.report(NMI_DBSCAN=nmi_dbscan, NMI_HDBSCAN=nmi_hdbscan, NMI_MD=nmi_md,
                    RI_DBSCAN=ri_hdbscan, RI_HDBSCAN=ri_hdbscan, RI_MD=ri_md,
                    BIC = bic, Silhouette_hdbscan=silhouette_hdbscan, Silhouette_md=silhouette_md)

        with open(log_folder, "a+") as f:
            print("{}_{}".format(config["lamb"], config["beta"]), file=f)
            print("{}\t{}\t{}".format(ri_dbscan, ari_dbscan, nmi_dbscan), file=f)
            print("{}\t{}\t{}".format(ri_hdbscan, ari_hdbscan, nmi_hdbscan), file=f)
            print("{}\t{}\t{}".format(ri_md, ari_md, nmi_md), file=f)
            print("{}\t{}\t{}\t{}\t{}".format(bic, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters, X.shape[0]), file=f)
            print("{}\t{}".format(silhouette_hdbscan, silhouette_md), file=f)

    """
    Simple test:

    _config = {
        "width" : 64,
        "stride" : 32,
        "lamb" : 5e-3,
        "beta" : 1e-2
    }

    train_TGMRF(_config)
    """

    with open(log_folder, "a+") as f:
        print("\n lamb_beta \n", file=f)

    analysis = tune.run(
        train_TGMRF,
        config={
                "lamb": tune.grid_search([i*0.01 for i in range(0, 13)]),
                "beta": tune.grid_search([i*0.01 for i in range(0, 13)])
                })

    result_folder = "./result/" + dataset_name + "/best_config.txt"
    

    with open(result_folder, "a+") as f:
        print("Best config of HDBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_HDBSCAN", mode="max"), file=f)

    print("Best config of HDBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_HDBSCAN", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of DBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_DBSCAN", mode="max"), file=f)

    print("Best config of DBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_DBSCAN", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of Multi-Density (TGMRF): ", analysis.get_best_config(metric="RI_MD", mode="max"), file=f)

    print("Best config of Multi-Density (TGMRF): ", analysis.get_best_config(metric="RI_MD", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of BIC (TGMRF): ", analysis.get_best_config(metric="BIC", mode="min"), file=f)

    print("Best config of BIC (TGMRF): ", analysis.get_best_config(metric="BIC", mode="min"))

    with open(result_folder, "a+") as f:
        print("Best config of Silhouette_hdbscan (TGMRF): ", analysis.get_best_config(metric="Silhouette_hdbscan", mode="max"), file=f)

    print("Best config of Silhouette_hdbscan (TGMRF): ", analysis.get_best_config(metric="Silhouette_hdbscan", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of Silhouette_md (TGMRF): ", analysis.get_best_config(metric="Silhouette_md", mode="max"), file=f)

    print("Best config of Silhouette_md (TGMRF): ", analysis.get_best_config(metric="Silhouette_md", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    df.to_csv("./result/" + dataset_name + "/parameter_search.csv")