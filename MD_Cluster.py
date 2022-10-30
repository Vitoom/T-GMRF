#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1 22:14:30 2020

@author: vito
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from Solver.TGMRF_solver import TGMRF_solver
from time import time

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if sum(abs(x_new - x_core)) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

class MD_Cluster:
    """
    The implementation is multi-density based clustering algorithm.

    Parameters
    ----------
    """
    def __init__(self, diff_threshold=0.0015, slope_threshold=0.015,k=10,k_dis_low=1e-12,k_dis_high=1e12):
        self.diff_threshold = diff_threshold
        self.slope_threshold = slope_threshold
        self.k = k
        self.select_portion = 0.3 # 0.25
        self.cluster_bag = []
        self.radiuses = None
        self.C_trans_dump = None
        self.cluster_result_dump = None
        self.k_dis_low = k_dis_low
        self.k_dis_high = k_dis_high

    def Infection_Point(self, dis_list, i, j):
        stack = []
        
        radiuses_index = []
        history = []
        stack.append((i, j))
        
        while(len(stack) != 0):
            left, right = stack.pop()
            #print(left, right)
            if right - left <= 1:
                continue
            r = -1
            diff = 1e30
            left_slope = 1e30
            right_slope = 1e30
            for k in range(left+1, right):
                left_slope = (dis_list[k] - dis_list[left]) / (k-left)
                right_slope = (dis_list[right] - dis_list[k]) / (right-k)
                #print(right_slope)
                if abs(left_slope - right_slope) < diff:
                    diff = abs(left_slope - right_slope)
                    _left = left_slope
                    _right = right_slope
                    r = k
                history.append((r, _left, _right, diff))
                #print(r, "\t", left_slope, "\t", right_slope, "\t", diff)
            if abs(diff) < self.diff_threshold and abs(_left) < self.slope_threshold and abs(_right) < self.slope_threshold:
                print(r, "left slope:\t", _left, "right slope:\t", _right, "slope diff:\t", diff)
                radiuses_index.append(r)
            stack.append((left, r-1))
            stack.append((r+1, right))

        radiuses_index = sorted(radiuses_index)
        return radiuses_index

    def get_radiuses(self, distance):
        k_dis = []
        for i in range(distance.shape[0]):
            _dis = distance[i,:].copy()
            _dis.sort()
            k_dis.append(_dis[self.k])
        k_dis = sorted(k_dis)
        radiuses_index = self.Infection_Point(k_dis, 0, len(k_dis) - 1)

        # densing adjacent radiuses
        error_list = []
        for i in range(1, len(radiuses_index)):
            error = radiuses_index[i] - radiuses_index[i-1]
            error_list.append(error)
        _eps = (max(radiuses_index) - min(radiuses_index)) * self.select_portion
        
        true_radiuses_index = []
        start = radiuses_index[0]
        true_radiuses_index.append(start)
        error_shift = 0
        for i in range(len(error_list)):
            error_shift = error_shift + error_list[i]
            if error_shift > _eps:
                true_radiuses_index.append(int(start + (radiuses_index[i+1] - start) * 0.8))
                error_shift = 0
                start = radiuses_index[i+1]

        # true_radiuses_index.append(int((start + radiuses_index[-1])/2))

        true_radiuses_index = np.array(true_radiuses_index)
        
        true_radiuses_index = true_radiuses_index[[k_dis[ele] > self.k_dis_low and k_dis[ele] < self.k_dis_high for ele in true_radiuses_index]]
        
        radiuses_value = [k_dis[i] for i in true_radiuses_index]

        if True:
            plot_radiuses_index = [len(k_dis) - i for i in true_radiuses_index]

            plt.plot(k_dis[:len(k_dis)][::-1], linewidth=6)
            plt.plot(plot_radiuses_index, radiuses_value, 'r^', markersize=12)

            eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 8) # 8 for BasicMotions
            plt.axhline(y=eps, color='y', linestyle='-')
            fig = plt.gcf()
            fig.savefig('MDDBSCAN_density.pdf', format='pdf', bbox_inches='tight')
    
        return radiuses_value

    def fit_predict(self, C_trans):

        # Compute similarity distance matrix
        distance = pairwise_distances(C_trans, metric="l1")

        true_radiuses = self.get_radiuses(distance)

        true_radiuses = sorted(true_radiuses)

        self.radiuses = true_radiuses

        print("Radiuses:", true_radiuses)

        clustering_assign = np.full(C_trans.shape[0], -1)

        C_trans_iter = C_trans.copy()

        stride = 0
        select_index = np.array(list(range(C_trans.shape[0])))
        
        if C_trans_iter.shape[0] in [80]:
            _min_samples = 4
        else:
            _min_samples = 3

        for radius in true_radiuses:
            dbscan = DBSCAN(eps=radius, min_samples=_min_samples, metric="l1")

            db_clustering = dbscan.fit_predict(C_trans_iter)
            self.cluster_bag.append(dbscan)

            select = db_clustering != -1
            remain = db_clustering == -1
            select_index_filter_noise = select_index[select]

            clustering_assign[select_index_filter_noise] = db_clustering[select] + stride

            select_index = select_index[remain]
            C_trans_iter = C_trans_iter[remain]
            if len(C_trans_iter) < 1:
                break
            stride += 100

        clustering_result = pd.Series(clustering_assign).astype('category').cat.codes

        self.C_trans_dump = C_trans
        self.cluster_result_dump = clustering_result

        return clustering_result

    def fit(self, C_trans):

        self.fit_predict(C_trans)

    def predict(self, C_trans):
        
        dist = pairwise_distances(C_trans, self.C_trans_dump, metric="l1")

        index = dist.argmin(axis=1)
        
        clustering_result = [self.cluster_result_dump[i] for i in index]

        return np.array(clustering_result)
