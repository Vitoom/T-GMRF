#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:19:39 2020

@author: vito
"""
import numpy as np
# import pandas as pd
# import pickle as pkl
# import os
# import seaborn as sns

# from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import cosine
from fastdtw import fastdtw
# from sklearn.cluster import DBSCAN
# from hdbscan.hdbscan_ import HDBSCAN
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import normalized_mutual_info_score
# from tqdm import tqdm
#from CSPCA import CSPCA

#import numpy as np
from sklearn import preprocessing
from dtw import dtw


class CSPCA:
    """
    The implementation is covariance sequence based principal component analysis(CSPCA).
    
    Parameters
    ----------
    epsilon: int, optinal(default=95)
             Cumulative energy content threshold  
    """
    
    def __init__(self, epsilon=95, width=10, stride=1, method=1, dtw_ = "common", measure="euclidean"):
        self.epsilon = epsilon
        self.width = width
        self.stride = stride
        self.method = method
        self.dtw = dtw_
        self.measure = measure
        
    def fit_transform(self, X):
        """
        Compute clustering and transform X to similarity distance matrix.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, m_lengths, l_features]
            New data to transform.
            
        Returns
        -------
        distance : array, shape [n_samples, n_samples]
                 similarity distance matrix.
        """
        if self.method == 1:
            # Compute covariance matrix for every MTS(multivariaten time series) 
            n_samples = X.shape[0]
            l_features = X.shape[1]
            C = np.zeros((l_features * l_features, n_samples))
            for i in range(n_samples):
                C[:, i] = np.cov(X[i]).reshape(-1)
            
            # normalizing C
            """
            # worsen performance
            quantile_transformer = preprocessing.QuantileTransformer(
                    output_distribution='normal', random_state=0)
            C = quantile_transformer.fit_transform(C)
            """
            C = preprocessing.normalize(C, norm='l2')
            
            # Covariance of C
            Sigma_c = np.cov(C)
            
            # Run SVD algorithm onto covariance matrix of C
            u, s, vh = np.linalg.svd(Sigma_c, full_matrices=True)
            
            # According to the energy content threshold, select the first k
            totally_variance = sum(s)
            for i in range(n_samples, 0, -1):
                if sum(s[:i])/totally_variance*100 < self.epsilon:
                    k = i + 1
                    break
            
            # Reconstruction of Sigma_c
            C_trans = np.dot(C.T, u[:, :k])
            
            # Compute similarity distance matrix
            if self.measure == "euclidean":
                distance_computer = lambda x, y: sum((x-y)**2)**0.5
            distance = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    distance[i][j] = distance[j][i] = distance_computer(C_trans[i, :], C_trans[j, :])
            
            return distance, C_trans
        
        if self.method == 2:
            # Compute covariance matrix for every MTS(multivariaten time series) 
            n_samples = X.shape[0] 
            m_lengths = X.shape[2]
            l_features = X.shape[1]
            s_windows = int((m_lengths - self.width) / self.stride + 1)
            C = np.zeros((l_features * l_features * s_windows,  n_samples))
            cov_matrix_len = l_features * l_features
            for i in range(n_samples):
                for j in range(s_windows):
                    C[j * cov_matrix_len: (j + 1) * cov_matrix_len, i] = np.cov(X[i, :, self.stride * j: self.width + self.stride * j]).reshape(-1)
            
            # normalizing C
            """
            # worsen performance
            quantile_transformer = preprocessing.QuantileTransformer(
                    output_distribution='normal', random_state=0)
            C = quantile_transformer.fit_transform(C)
            """
            C = preprocessing.normalize(C, norm='l2')
            
            # Covariance of C
            Sigma_c = np.cov(C)
            
            # Run SVD algorithm onto covariance matrix of C
            u, s, vh = np.linalg.svd(Sigma_c, full_matrices=True)
            
            # According to the energy content threshold, select the first k
            totally_variance = sum(s)
            for i in range(n_samples, 0, -1):
                if sum(s[:i])/totally_variance*100 < self.epsilon:
                    k = i + 1
                    break
            
            # Reconstruction of Sigma_c
            C_trans = np.dot(C.T, u[:, :k])
            
            # Compute similarity distance matrix
            if self.measure == "euclidean":
                distance_computer = lambda x, y: sum((x-y)**2)**0.5
            distance = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    distance[i][j] = distance[j][i] = distance_computer(C_trans[i, :], C_trans[j, :])
            
            return distance, C_trans
        
        if self.method == 3:
            # Compute covariance matrix for every MTS(multivariaten time series) 
            n_samples = X.shape[0] 
            m_lengths = X.shape[1]
            l_features = X.shape[2]
            s_windows = int((m_lengths - self.width) / self.stride + 1)
            C = np.zeros((l_features * l_features, s_windows * n_samples))
            for i in range(n_samples):
                for j in range(s_windows):
                    C[:, i*s_windows + j] = np.cov(X[i, self.stride * j: self.width + self.stride * j].T).reshape(-1)
            
            # normalizing C
            """
            # worsen performance
            quantile_transformer = preprocessing.QuantileTransformer(
                    output_distribution='normal', random_state=0)
            C = quantile_transformer.fit_transform(C)
            """
            C = preprocessing.normalize(C, norm='l2')
            
            # Covariance of C
            Sigma_c = np.cov(C)
            
            # Run SVD algorithm onto covariance matrix of C
            u, s, vh = np.linalg.svd(Sigma_c, full_matrices=True)
            
            # According to the energy content threshold, select the first k
            totally_variance = sum(s)
            for i in range(n_samples, 0, -1):
                if sum(s[:i])/totally_variance*100 < self.epsilon:
                    k = i + 1
                    break
            
            # Reconstruction of Sigma_c
            C_trans = np.dot(C.T, u[:, :k])
            
            # Reconstruction of covariance sequence
            C_seq = C_trans.reshape(-1, s_windows, C_trans.shape[1])
            
            # Compute similarity distance matrix
            if self.measure == "euclidean":
                distance_computer = lambda x, y: sum((x-y)**2)**0.5
            distance = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if(self.dtw == "common"):
                        tmp, _, _, _ = dtw(C_seq[i], C_seq[j], dist=distance_computer)
                        distance[i][j] = distance[j][i] = tmp
                    else:
                        tmp, _ = fastdtw(C_seq[i], C_seq[j], dist=distance_computer)
                        distance[i][j] = distance[j][i] = tmp
            
            return distance, C_seq