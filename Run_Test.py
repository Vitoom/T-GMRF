#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:14:30 2020

@author: vito
"""
import os
import pickle as pkl
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from tslearn.clustering import KShape

from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.measures import rand_score

from Tools.Dataset_Reader import ArffDataset_Generate
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn.metrics import pairwise_distances

X_train, Y_train, X_test, Y_test = ArffDataset_Generate("BasicMotions")

dump_file = "./dump/C_trans_dump.pkl"

if not os.path.exists(dump_file):

    # Combining train and test data to extract T-GMRF features
    clf = TGMRF(width=50, stride=50, maxIters=30, lamb=1e-2, beta=1e-2)
    C_trans, C_trans_train, C_trans_test = clf.fit(X_train, X_test)

    output = open(dump_file, 'wb')
    pkl.dump((C_trans, C_trans_train, C_trans_test), output)
else:
    output = open(dump_file, 'rb')
    C_trans, C_trans_train, C_trans_test = pkl.load(output)

output.close()

# Train multi-density clustering algorithm
clustering = MD_Cluster(diff_threshold=0.003, slope_threshold=0.01)
_clustering = clustering.fit_predict(C_trans)

clustering_predict = clustering.predict(C_trans_test)

ri_md = rand_score(clustering_predict, Y_test)
nmi_md = normalized_mutual_info_score(clustering_predict, Y_test)

print("Rand Score (TGMRF + Multi-density):\t{}".format(ri_md))
print("NMI (TGMRF + Multi-density):\t{}\n".format(nmi_md))

# Pure DBSCAN
distance = pairwise_distances(C_trans, metric="l1")
_eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 10)
clustering = DBSCAN(eps=_eps, min_samples=3, metric="l1")

clustering_db = clustering.fit_predict(C_trans)

ri_db = rand_score(clustering_db[-40:], Y_test)
nmi_db = normalized_mutual_info_score(clustering_db[-40:], Y_test)

print("Rand Score (TGMRF + Pure DBSCAN):\t{}".format(ri_db))
print("NMI (TGMRF + Pure DBSCAN):\t{}\n".format(nmi_db))

# K-means

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(C_trans)
clustering_kmeans =  kmeans.predict(C_trans_test)

ri_kmeans = rand_score(clustering_kmeans, Y_test)
nmi_kmeans = normalized_mutual_info_score(clustering_kmeans, Y_test)

print("Rand Score (TGMRF + KMeans):\t{}".format(ri_kmeans))
print("NMI (TGMRF + KMeans):\t{}\n".format(nmi_kmeans))

# K-Shape
X = np.concatenate((X_train, X_test), axis=0)
_X = X.transpose(0, 2, 1)
_X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(_X)
ks = KShape(n_clusters=4, n_init=6, random_state=0)

clustering_result = ks.fit(_X)
clustering_ks = ks.predict(_X[-40:])

ri_ks = rand_score(clustering_ks, Y_test)
nmi_ks = normalized_mutual_info_score(clustering_ks, Y_test)

print("Rand Score (KShape):\t{}".format(ri_ks))
print("NMI (KShape):\t{}".format(nmi_ks))

"""
Rand Score (TGMRF + Multi-density):     0.9141025641025641
NMI (TGMRF + Multi-density):    0.8426223435220391

Rand Score (TGMRF + Pure DBSCAN):       0.8807692307692307
NMI (TGMRF + Pure DBSCAN):      0.7797921224906507

Rand Score (TGMRF + KMeans):    0.8602564102564103
NMI (TGMRF + KMeans):   0.8293595724997522

Rand Score (KShape):    0.8166666666666667
NMI (KShape):   0.600817211326205
"""






