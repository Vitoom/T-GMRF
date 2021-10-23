#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:19:39 2020

@author: vito
"""
import numpy as np
# import pandas as pd
# import pickle as pkl
import os
# import seaborn as sns

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
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
from math import sqrt
from .SBD import SBD

from .DPC.DPC import DPC
import math

import pickle as pkl
from fcmeans import FCM

from sklearn import metrics

try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa 

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluation(prediction, label):
    acc = cluster_acc(label, prediction)
    nmi = metrics.normalized_mutual_info_score(label, prediction)
    ari = metrics.adjusted_rand_score(label, prediction)
    ri = rand_index_score(label, prediction)
    print((acc, nmi, ari, ri))
    return ri, nmi, acc

class FCFW:
    """
    The implementation is covariance sequence based principal component analysis(CSPCA).
    
    Parameters
    ----------
    epsilon: int, optinal(default=95)
             Cumulative energy content threshold  
    """
    
    def __init__(self, epsilon=95, width=10, stride=1, method=1, measure="euclidean", shrink_scale=1/10, pkl_updating=False):
        self.epsilon = epsilon
        self.width = width
        self.stride = stride
        self.method = method
        self.measure = measure
        self.shrink_scale = shrink_scale
        self.pkl_updating = pkl_updating
        self.X = None
        
    def compute_weights(self):
        weights = np.zeros(self.X.shape[1])
        
        _max = [self.X[:,i,:].max() for i in range(self.X.shape[1])]
        _min = [self.X[:,i,:].min() for i in range(self.X.shape[1])]
        error = [_max[i] - _min[i] for i in range(self.X.shape[1])]
        denominator = sum(error)
        
        weights = np.array([(_max[i] - _min[i]) / denominator for i in range(self.X.shape[1])])
        
        return weights
    
    def compute_dtw(self):
        # save or load datasetruct for time-saving --- dtw
        
        if self.measure == "euclidean":
            _metric = euclidean
        elif self.measure == "cosine":
            _metric = cosine
        
        dump_file = "../dump/" + "shrink_scale_{:.5f}".format(self.shrink_scale) + "_dtw.pkl"
        if self.pkl_updating or not os.path.exists(dump_file):
            
            num_instance = self.X.shape[0]
            
            # datastruct to load dtw
            _dtw = np.zeros((num_instance, num_instance))
            
            # computing dtw
            for i in range(num_instance):
                for j in range(i+1, num_instance):
                    # for k in range(45):
                    #    tmp, _ = fastdtw(X[i,k], X[j,k], dist=_metric)
                    tmp, _ = fastdtw(self.X[i].T, self.X[j].T, dist=_metric)
                    _dtw[i, j] = sqrt(tmp)
                    _dtw[j, i] = _dtw[i, j]
            
            output = open(dump_file, 'wb')
            pkl.dump(_dtw, output)
        else:
            output = open(dump_file, 'rb')
            _dtw = pkl.load(output)
        output.close()
        
        return _dtw
        
    def comput_SBD(self):
        
        _sbd = np.zeros((self.X.shape[0], self.X.shape[0], self.X.shape[1]))
        
        num_instance = self.X.shape[0]

        for k in range(self.X.shape[1]):
            for i in range(num_instance):
                for j in range(i+1, num_instance):
                    # for k in range(45):
                    #    tmp, _ = fastdtw(X[i,k], X[j,k], dist=_metric)
                    _sbd[i, j, k] = SBD(self.X[i, k], self.X[j, k])
                    _sbd[j, i, k] = _sbd[i, j, k] # k: dimension
                    
        return _sbd
        
    def fit_transform(self, X, n_class=6):
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
        
        self.X = X
        
        weights = self.compute_weights()
        
        _dtw = self.compute_dtw()
        
        _sbd = self.comput_SBD()
        
        dpc = DPC()
        centers = dpc.get_centers(self.X, n=n_class)
        
        manhattan = lambda a, b: sum(abs(a.reshape(-1) - b.reshape(-1)))
        
        F_dtw = np.zeros((X.shape[0], len(centers)))
        
        n = 0
        for center in centers:
            F_dtw[:, n] = 1 / (np.sqrt(_dtw[:, center]) * np.sqrt(1 / (_dtw[:, centers] + 1e-12)).sum(axis=1) + 1e-12)
            F_dtw[centers, n] = [int(center == centers[i]) for i in range(len(centers))]
            n = n + 1
        
        F_sbd = np.zeros((X.shape[0], len(centers), X.shape[1])) # [n_instance, n_classes, n_dimension]
        for k in range(X.shape[1]):
            n = 0
            for center in centers:
                F_sbd[:, n, k] = 1 / (np.sqrt(_sbd[:, center, k]) * np.sqrt(1 / (_sbd[:, centers, k] + 1e-12)).sum(axis=1) + 1e-12)
                F_sbd[centers, n, k] = [int(center == centers[i]) for i in range(len(centers))]
                n = n + 1
            F_sbd[:, :, k] = F_sbd[:, :, k] * weights[k]
            
        F_sbd_combine = F_sbd.sum(axis = 2)
        
        F = (F_dtw + F_sbd_combine) / 2
        
        return F
        
    
if __name__ == "__main__":
    os.chdir("/home/djh/dwx/MTS_Cluster/Models")

    shrink_scale = 0.000816
    pkl_updating = False
    dump_file = "../dump/dataset_"+ "shrink_scale_{:.5f}".format(shrink_scale) + ".pkl"
    output = open(dump_file, "rb")
    X, Y = pkl.load(output)
        
    num_instance = X.shape[0]
    
    fcfw = FCFW(shrink_scale=shrink_scale, pkl_updating=pkl_updating)
    _features = fcfw.fit_transform(X)
    
    features = _features + 1e-2
    
    fcm = FCM(n_clusters=3)
    fcm.fit(features)
    
    # fcm_centers = fcm.centers
    fcm_labels = fcm.predict(features)
    
    evaluation(fcm_labels, Y)