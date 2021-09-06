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

class MD_Cluster:
    """
    The implementation is multi-density based clustering algorithm.

    Parameters
    ----------
    """
    def __init__(self, diff_threshold=0.0015, slope_threshold=0.015):
        self.diff_threshold = diff_threshold
        self.slope_threshold = slope_threshold
        self.k = 20
        self.select_portion = 0.5

    def Infection_Point(self, dis_list, i, j):
        stack = []
        radiuses = []
        stack.append((i, j))
        dis_list = sorted(dis_list)
        while(len(stack) != 0):
            left, right = stack.pop()
            #print(left, right)
            if right - left > 1:
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
                        r = k
                    print(left_slope, "\t", right_slope, "\t", diff)
                if diff < self.diff_threshold and left_slope < self.slope_threshold and right_slope < self.slope_threshold:
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
            _eps = (max(error_list) - min(error_list)) * self.select_portion
        
        true_radiuses = []
        start = radiuses[0]
        true_radiuses.append(start)
        for i in range(len(error_list)):
            if error_list[i] > _eps:
                true_radiuses.append(radiuses[i+1])
    
        return true_radiuses

    def fit_predict(self, distance):
        """
        Only precomputed distance is supported since different distance measures for different features 
        """
        true_radiuses = self.get_radiuses(distance)

        true_radiuses = sorted(true_radiuses)

        clustering_assign = np.full(len(distance), -1)

        distance_iter = distance
        stride = 0
        select_index = np.array(list(range(len(distance))))
        for radius in true_radiuses:
            db_clustering = DBSCAN(eps=radius, min_samples=3).fit_predict(distance_iter) + stride
            select = db_clustering != -1
            remain = db_clustering == -1
            select_index_filter_noise = select_index[select]

            clustering_assign[select_index_filter_noise] = db_clustering[select]

            select_index = select_index[remain]
            distance_iter = distance_iter[remain][:,remain]

            stride += 100

        clustering_result = pd.Series(clustering_assign).astype('category').cat.codes

        return clustering_result






