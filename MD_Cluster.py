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

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class MD_Cluster:
    """
    The implementation is multi-density based clustering algorithm.

    Parameters
    ----------
    """
    def __init__(self, diff_threshold=0.0015, slope_threshold=0.015):
        self.diff_threshold = diff_threshold
        self.slope_threshold = slope_threshold
        self.k = 10
        self.select_portion = 0.3
        self.cluster_bag = []
        self.radiuses = None

    def Infection_Point(self, dis_list, i, j):
        stack = []
        radiuses = []
        history = []
        stack.append((i, j))
        dis_list = sorted(dis_list)
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
                print(r, "\t", left_slope, "\t", right_slope, "\t", diff)
            if abs(diff) < self.diff_threshold and abs(left_slope) < self.slope_threshold and abs(right_slope) < self.slope_threshold:
                radiuses.append(dis_list[r])
            stack.append((left, r-1))
            stack.append((r+1, right))

        radiuses = sorted(radiuses)
        print(radiuses)
        return radiuses

    def get_radiuses(self, distance):
        k_dis = []
        for i in range(distance.shape[0]):
            _dis = distance[i,:].copy()
            _dis.sort()
            k_dis.append(_dis[self.k])
        
        radiuses = self.Infection_Point(k_dis, 0, len(k_dis) - 1)

        error_list = []
        for i in range(1, len(radiuses)):
            error = radiuses[i] - radiuses[i-1]
            error_list.append(error)
        _eps = (max(radiuses) - min(radiuses)) * self.select_portion
        
        true_radiuses = []
        start = radiuses[0]
        true_radiuses.append(start)
        for i in range(len(error_list)):
            if error_list[i] > _eps:
                true_radiuses.append(int((start + radiuses[i])/2))
                start = radiuses[i+1]
        
        true_radiuses.append(int((start + radiuses[-1])/2))

        true_radiuses = list(set(true_radiuses))

        plot_radiuses_value = [k_dis[i] for i in true_radiuses]
        plot_radiuses = [len(k_dis) - i for i in true_radiuses]

        plt.plot(k_dis[:len(k_dis)][::-1], linewidth=6)
        plt.plot(plot_radiuses, plot_radiuses_value, 'r^', markersize=12)

        eps = np.percentile(distance.reshape(-1)[distance.reshape(-1) != 0], 9)
        plt.axhline(y=eps, color='y', linestyle='-')
        fig = plt.gcf()
        fig.savefig('MDDBSCAN_density.pdf', format='pdf', bbox_inches='tight')
    
        return true_radiuses

    def fit_predict(self, C_trans):

        n_samples = C_trans.shape[0]

        # Compute similarity distance matrix
        distance = pairwise_distances(C_trans, metric="l1")

        true_radiuses = self.get_radiuses(distance)

        true_radiuses = sorted(true_radiuses)

        self.radiuses = true_radiuses

        clustering_assign = np.full(C_trans.shape[0], -1)

        C_trans_iter = C_trans.copy()

        stride = 0
        select_index = np.array(list(range(C_trans.shape[0])))

        for radius in true_radiuses:
            dbscan = DBSCAN(eps=radius, min_samples=3, metric="l1")
            db_clustering = dbscan.fit_predict(C_trans_iter)
            self.cluster_bag.append(dbscan)

            select = db_clustering != -1
            remain = db_clustering == -1
            select_index_filter_noise = select_index[select]

            clustering_assign[select_index_filter_noise] = db_clustering[select]

            select_index = select_index[remain]
            C_trans_iter = C_trans_iter[remain]

            stride += 100

        clustering_result = pd.Series(clustering_assign).astype('category').cat.codes

        return clustering_result

    def fit(self, C_trans):

        self.fit_predict()

    def predict(self, C_trans):
        n_samples = C_trans.shape[0]

        C_trans_iter = C_trans.copy()

        stride = 0
        select_index = np.array(list(range(C_trans.shape[0])))

        clustering_assign = np.full(C_trans.shape[0], -1)

        for dbscan in self.cluster_bag:
            db_clustering = dbscan.predict(C_trans_iter)
            self.cluster_bag.append(dbscan)

            select = db_clustering != -1
            remain = db_clustering == -1
            select_index_filter_noise = select_index[select]

            clustering_assign[select_index_filter_noise] = db_clustering[select]

            select_index = select_index[remain]
            C_trans_iter = C_trans_iter[remain]

            stride += 100

        clustering_result = pd.Series(clustering_assign).astype('category').cat.codes

        return clustering_result






