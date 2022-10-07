#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:14:30 2020

@author: vito
"""
import os
import pickle as pkl
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from Tools.generate_synthetic_data_tool import Fake_Dataset
from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.RI import rand_score
from sklearn.metrics import pairwise_distances

dataset_name = "Fake_192_300_4" # "Fake_30_30_4"

X, Y = Fake_Dataset(dataset_name) # "Fake_192_300_4"

X = X.transpose((0, 2, 1))

dump_file = "./dump/" + dataset_name + "/TGMRF.pkl"
if not os.path.exists(dump_file):
    clf = TGMRF(epsilon=50, width=10, stride=10, maxIters=80, lamb=5e-3, beta=5e-3) # maxIters=80
    C_trans, duration, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = clf.fit_transform(X)
    
    output = open(dump_file, 'wb')
    pkl.dump(C_trans, output)
else:
    output = open(dump_file, 'rb')
    C_trans = pkl.load(output)
output.close()

# Compute similarity distance matrix
distance = pairwise_distances(C_trans, metric="l1")

clustering = MD_Cluster(diff_threshold=0.003, slope_threshold=0.015)
clustering_result_md = clustering.fit_predict(distance)

_eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 10)
clustering_dbscan = DBSCAN(eps=_eps, min_samples=2, metric="precomputed").fit_predict(distance)

clustring_kmeans = KMeans(n_clusters=len(set(Y)), random_state=0).fit_predict(C_trans)

ri_md = rand_score(clustering_result_md, Y)
ri_db = rand_score(clustering_dbscan, Y)
ri_km = rand_score(clustring_kmeans, Y)

print("Rand Score (Multi-density):\t{}".format(ri_md))
print("Rand Score (DBSCAN):\t{}".format(ri_db))
print("Rand Score (kmeans):\t{}".format(ri_km))





