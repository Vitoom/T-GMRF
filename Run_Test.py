#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:14:30 2020

@author: vito
"""
from curses import doupdate
import os
import pickle as pkl
from tokenize import Double
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from tslearn.clustering import KShape

from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.RI import rand_score

from Tools.Dataset_Reader import Get_Dataset
from Tools.Root_Path import Root_Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn.metrics import pairwise_distances

from Plot_Embedding import Plot_Embedding

dataset_name = "HAR"

print("Processing {0}".format(dataset_name))

os.chdir(Root_Path)

parameters = pd.read_csv('./Parameters.csv', sep=',', index_col=0)

parameters = parameters.astype({"width": int, "stride": int, "lamb": float, "beta": float, "diff_threshold": float, "slope_threshold": float})

parameter = parameters.loc[dataset_name]

X_train, Y_train, X_test, Y_test = Get_Dataset(dataset_name)

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

dump_file = f"./dump/C_trans_dump_{dataset_name}.pkl"

if not os.path.exists(dump_file):

    # Combining train and test data to extract T-GMRF features
    clf = TGMRF(width=parameter["width"].astype(int), stride=parameter["stride"].astype(int), lamb=parameter["lamb"], beta=parameter["beta"], maxIters=int(parameter["maxIters"]), verbose_ADMM=False,dimension_reduce=parameter["dimension_reduce"].astype(bool), epsilon=parameter["CumulativeEnergySaving"].astype(int),dataset_name=dataset_name,use_dump=True,maxIters_ADMM=parameter["maxIters_ADMM"].astype(int))
    C_trans, C_trans_train, C_trans_test = clf.fit(X_train, X_test)

    output = open(dump_file, 'wb')
    pkl.dump((C_trans, C_trans_train, C_trans_test), output)
else:
    output = open(dump_file, 'rb')
    C_trans, C_trans_train, C_trans_test = pkl.load(output)

output.close()

# Train multi-density clustering algorithm
clustering = MD_Cluster(diff_threshold=parameter["diff_threshold"], slope_threshold=parameter["slope_threshold"], k=int(parameter["k_nearest"]), k_dis_low=parameter["k_dis_low"].astype(float), k_dis_high=parameter["k_dis_high"].astype(float))
clustering.fit(C_trans)

clustering_predict = clustering.predict(C_trans_test)
_clustring_predict = clustering.predict(C_trans)

ri_md = rand_score(clustering_predict, Y_test)
nmi_md = normalized_mutual_info_score(clustering_predict, Y_test)

print("Rand Score (TGMRF + Multi-density):\t{}".format(ri_md))
print("NMI (TGMRF + Multi-density):\t{}".format(nmi_md))

# Pure DBSCAN
distance = pairwise_distances(C_trans, metric="l1")
_eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 9) # 8 for BasicMotions
clustering = DBSCAN(eps=_eps, min_samples=3, metric="l1")

clustering_db = clustering.fit_predict(C_trans)

ri_db = rand_score(clustering_db[-len(Y_test):], Y_test)
nmi_db = normalized_mutual_info_score(clustering_db[-len(Y_test):], Y_test)

print("Rand Score (TGMRF + Pure DBSCAN):\t{}".format(ri_db))
print("NMI (TGMRF + Pure DBSCAN):\t{}".format(nmi_db))

# K-means
kmeans = KMeans(n_clusters=len(set(Y_test)), random_state=5)
kmeans.fit(C_trans)
clustering_kmeans =  kmeans.predict(C_trans_test)
_clustring_kmeans = kmeans.predict(C_trans)

ri_kmeans = rand_score(clustering_kmeans, Y_test)
nmi_kmeans = normalized_mutual_info_score(clustering_kmeans, Y_test)

print("Rand Score (TGMRF + KMeans):\t{}".format(ri_kmeans))
print("NMI (TGMRF + KMeans):\t{}".format(nmi_kmeans))

# K-Shape
"""
X = np.concatenate((X_train, X_test), axis=0)
_X = X.transpose(0, 2, 1)
_X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(_X)

ks = KShape(n_clusters=len(set(Y_test)), n_init=6, random_state=5)

ks.fit(_X)
clustering_ks = ks.predict(_X[-len(Y_test):])

ri_ks = rand_score(clustering_ks, Y_test)
nmi_ks = normalized_mutual_info_score(clustering_ks, Y_test)

print("Rand Score (KShape):\t{}".format(ri_ks))
print("NMI (KShape):\t{}".format(nmi_ks))
"""

Plot_Embedding(distance, len(Y_train), np.concatenate((Y_train, Y_test)), _clustring_kmeans, clustering_db, _clustring_predict)







