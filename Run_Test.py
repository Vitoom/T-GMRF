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

from TGMRF import TGMRF
from MD_Cluster import MD_Cluster
from Measures.measures import rand_score

from Tools.Dataset_Reader import ArffDataset_Generate

X_train, Y_train, X_test, Y_test = ArffDataset_Generate("BasicMotions")

X = np.concatenate((X_train,X_test), axis=0)
Y = np.concatenate((Y_train,Y_test), axis=0)

clf = TGMRF(epsilon=50, width=10, stride=10, maxIters=80, lamb=5e-3, beta=5e-3) # maxIters=80
C_trans, duration, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = clf.fit_transform(X)

clustering = MD_Cluster(diff_threshold=0.0015, slope_threshold=0.015)
clustering_result_md = clustering.fit_predict(C_trans)

ri_md = rand_score(clustering_result_md, Y)

print("Rand Score (Multi-density):\t{}".format(ri_md))





