#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:06:09 2020

@author: vito
"""

import os
import numpy as np
import pandas as pd
import math
from ray import tune

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, pairwise_distances

from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.RI import rand_score
from Tools.Dataset_Reader import Get_Dataset
from Tools.Root_Path import Root_Path

os.chdir(Root_Path)

def Tuning_Hyperparametes(dataset_name):

    X, Y, _, _ = Get_Dataset(dataset_name)

    log_folder = Root_Path + "/result/" + dataset_name + "/config_performance.txt"
    directory = os.path.dirname(log_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    parameters = pd.read_csv(Root_Path + '/Parameters.csv', sep=',', index_col=0)

    parameters = parameters.astype({"width": int, "stride": int, "lamb": float, "beta": float, "diff_threshold": float, "slope_threshold": float})

    parameter = parameters.loc[dataset_name]

    def train_TGMRF(config):
        if config["width"] < config["stride"]:
            return
        
        clf = TGMRF(width=config["width"], stride=config["stride"], lamb=config["lamb"], beta=config["beta"], maxIters=int(parameter["maxIters"]), verbose_ADMM=False,dimension_reduce=parameter["dimension_reduce"].astype(bool), epsilon=config["CumulativeEnergySaving"],dataset_name=dataset_name)
        icspca, _, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = clf.fit_transform(X) 

        distance = pairwise_distances(icspca, metric="l1")
        _eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 9)
        clustering_dbscan = DBSCAN(eps=_eps, min_samples=3, metric="precomputed").fit(distance)

        clustering_MD = MD_Cluster(diff_threshold=parameter["diff_threshold"], slope_threshold=parameter["slope_threshold"], k=int(parameter["k_nearest"]), k_dis_low=parameter["k_dis_low"].astype(float), k_dis_high=parameter["k_dis_high"].astype(float))
        clustering_result_md = clustering_MD.fit_predict(icspca)

        shrink = 0.15
        bic = math.log(X.shape[0]) * numberOfParameters * shrink - 2 * math.log(aggregated_ll_Loss) # math.log(aggregated_ll_Loss + aggregated_penalty_loss)

        silhouette_dbscan = silhouette_score(distance, clustering_dbscan.labels_, metric="precomputed")
        silhouette_md = silhouette_score(distance, clustering_result_md, metric="precomputed")

        ri_dbscan = rand_score(clustering_dbscan.labels_, Y)
        ri_md = rand_score(clustering_result_md, Y)

        nmi_dbscan = normalized_mutual_info_score(clustering_dbscan.labels_, Y)
        nmi_md = normalized_mutual_info_score(clustering_result_md, Y)

        kmeans = KMeans(n_clusters=len(set(Y)), random_state=5)
        kmeans.fit(icspca)
        clustering_kmeans =  kmeans.predict(icspca)

        ri_kmeans = rand_score(clustering_kmeans, Y)
        nmi_kmeans = normalized_mutual_info_score(clustering_kmeans, Y)
        silhouette_kmeans = silhouette_score(distance, clustering_kmeans, metric="precomputed")

        tune.report(NMI_DBSCAN=nmi_dbscan, RI_DBSCAN=ri_dbscan, NMI_MD=nmi_md, RI_MD=ri_md,
                    RI_KMEANS=ri_kmeans, NMI_KMEANS=nmi_kmeans, BIC=bic, 
                    Silhouette_dbscan=silhouette_dbscan, Silhouette_md=silhouette_md,
                    Silhouette_kmeans=silhouette_kmeans)

        with open(log_folder, "a+") as f:
            print("{}_{}_{}_{}_{}".format(config["width"], config["stride"], config["CumulativeEnergySaving"], config["lamb"], config["beta"]), file=f)
            print("{}\t{}".format(ri_dbscan, nmi_dbscan), file=f)
            print("{}\t{}".format(ri_md, nmi_md), file=f)
            print("{}\t{}".format(ri_kmeans, nmi_kmeans), file=f)
            print("{}\t{}\t{}\t{}\t{}".format(bic, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters, X.shape[0]), file=f)
            print("{}\t{}\t{}".format(silhouette_dbscan, silhouette_md, silhouette_kmeans), file=f)

    with open(log_folder, "a+") as f:
        print("#" * 100, file=f)
    
    analysis = tune.run(
    train_TGMRF,
    config={
            "width": tune.grid_search([25, 30, 50, 60, 80, 103]), 
            "stride": tune.grid_search([25, 30, 50, 60, 80, 103]),
            "lamb": tune.grid_search([1e-1, 1e-2, 5e-2]),
            "beta": tune.grid_search([1e-1, 1e-2, 5e-2]),
            "CumulativeEnergySaving": tune.grid_search([10, 30, 50])
            })

    result_folder = Root_Path + "/result/" + dataset_name + "/best_config.txt"

    with open(result_folder, "a+") as f:
        print("Best config of BIC (TGMRF): ", analysis.get_best_config(metric="BIC", mode="min"), file=f)

    print("Best config of BIC (TGMRF): ", analysis.get_best_config(metric="BIC", mode="min"))

    with open(result_folder, "a+") as f:
        print("Best config of Silhouette_dbscan (TGMRF): ", analysis.get_best_config(metric="Silhouette_dbscan", mode="max"), file=f)

    print("Best config of Silhouette_dbscan (TGMRF): ", analysis.get_best_config(metric="Silhouette_dbscan", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of Silhouette_md (TGMRF): ", analysis.get_best_config(metric="Silhouette_md", mode="max"), file=f)

    print("Best config of Silhouette_md (TGMRF): ", analysis.get_best_config(metric="Silhouette_md", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of Silhouette_kmeans (TGMRF): ", analysis.get_best_config(metric="Silhouette_kmeans", mode="max"), file=f)

    print("Best config of Silhouette_kmeans (TGMRF): ", analysis.get_best_config(metric="Silhouette_kmeans", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of DBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_DBSCAN", mode="max"), file=f)

    print("Best config of DBSCAN (TGMRF): ", analysis.get_best_config(metric="RI_DBSCAN", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of Multi-Density (TGMRF): ", analysis.get_best_config(metric="RI_MD", mode="max"), file=f)

    print("Best config of Multi-Density (TGMRF): ", analysis.get_best_config(metric="RI_MD", mode="max"))

    with open(result_folder, "a+") as f:
        print("Best config of K-Means (TGMRF): ", analysis.get_best_config(metric="RI_KMEANS", mode="max"), file=f)

    print("Best config of K-Means (TGMRF): ", analysis.get_best_config(metric="RI_KMEANS", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    df.to_csv(Root_Path + "/result/" + dataset_name + "/parameter_search.csv")

Tuning_Hyperparametes("HAR")