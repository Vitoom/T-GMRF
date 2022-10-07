#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:14:30 2020

@author: vito
"""
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from Solver.TGMRF_solver import TGMRF_solver
from time import time
import os
import pickle as pkl

class TGMRF:
    """
    The implementation is Time-varying Gaussian Markov Random Feilds based clustering algorithm
    
    Parameters
    ----------
    """
    def __init__(self, epsilon=50, width=10, stride=1, maxIters=30, lr=0, lamb=1e-2, beta=1e-2, measure="euclidean", verbose=True, verbose_ADMM=False,dimension_reduce=True,dataset_name="Test",use_dump=False):
        self.epsilon = epsilon
        self.width = width
        self.stride = stride
        self.measure = measure
        self.maxIters = maxIters
        self.lr =lr
        self.lamb = lamb
        self.beta = beta
        self.verbose = verbose
        self.project_matrix = None
        self.initilizing = False
        self.verbose_ADMM = verbose_ADMM
        self.dimension_reduce = dimension_reduce
        self.dataset_name = dataset_name
        self.use_dump = use_dump
        
    def triangle_l_2_matrix_l(self, l):
        n = int((-1  + np.sqrt(1+ 8*l))/2)
        return n
        
    def upper2Full(self, a):
        n = self.triangle_l_2_matrix_l(a.shape[0])
        A = np.zeros([n,n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A
    
    def predict(self, X):
        """
        Fix the model and construct the project matrix
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, l_features, m_lengths]
            New data to transform.
            
        Returns
        -------
        C_trans : array, shape [n_samples, k]
                 Compacted vectors of T-GMRF after PCA
        """
        if not type(self.project_matrix) is np.ndarray:
            raise RuntimeError('Please fitting the model beforehand!')

        # Compute Time-varying Gaussian Markov Random Fields for every MTS (multivariaten time series) 
        n_samples = X.shape[0]
        m_lengths = X.shape[2]
        l_features = X.shape[1]
        s_windows = int((m_lengths - self.width) / self.stride + 1)
        self.C = np.zeros((int(l_features * (l_features + 1)  * s_windows / 2),  n_samples))
        cov_matrix_len = int(l_features * (l_features + 1) / 2)

        clf = TGMRF_solver(width=self.width, stride=self.stride, 
                  maxIters=self.maxIters, lr=self.lr, lamb=self.lamb, beta=self.beta, initilizing=self.initilizing, verbose_ADMM=self.verbose_ADMM)
        
        aggregated_ll_Loss = 0
        aggregated_penalty_loss = 0

        for i in tqdm(range(n_samples), ascii=True, desc="TGMRF"):
            ics, loss, ll_loss, penalty_loss, numberOfParameters = clf.fit(X[i].T)
            aggregated_ll_Loss += ll_loss
            aggregated_penalty_loss += penalty_loss
            for j in range(s_windows):
                self.C[j * cov_matrix_len: (j + 1) * cov_matrix_len, i] = ics[j]

        C_normalize = preprocessing.normalize(self.C, norm='l2')

        # Projecting the features
        C_trans = np.dot(C_normalize.T, self.project_matrix)

        return C_trans
    
    def fit_transform(self, X):
        """
        Transform X todistance matrix.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, l_features, m_lengths]
            New data to transform.
            
        Returns
        -------
        distance : array, shape [n_samples, n_samples]
                 similarity distance matrix.
        ...
                 Other useful data structure.
        """
        # Compute Time-varying Gaussian Markov Random Fields for every MTS (multivariaten time series) 
        n_samples = X.shape[0]
        l_features = X.shape[1]
        m_lengths = X.shape[2]
        s_windows = int((m_lengths - self.width) / self.stride + 1)
        self.C = np.zeros((int(l_features * (l_features + 1)  * s_windows / 2),  n_samples))
        cov_matrix_len = int(l_features * (l_features + 1) / 2)

        duration, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = 0, 0, 0, 0

        start = time()

        dump_file = f"./dump/{self.dataset_name}/T_GMRF_{self.dataset_name}_dump.pkl"

        if not os.path.exists(dump_file) or not self.use_dump:

            clf = TGMRF_solver(width=self.width, stride=self.stride, 
                    maxIters=self.maxIters, lr=self.lr, lamb=self.lamb, beta=self.beta, verbose_ADMM=self.verbose_ADMM)
            
            aggregated_ll_Loss = 0
            aggregated_penalty_loss = 0

            for i in tqdm(range(n_samples), ascii=True, desc="TGMRF"):
                ics, loss, ll_loss, penalty_loss, numberOfParameters = clf.fit(X[i].T)
                aggregated_ll_Loss += ll_loss
                aggregated_penalty_loss += penalty_loss
                for j in range(s_windows):
                    self.C[j * cov_matrix_len: (j + 1) * cov_matrix_len, i] = ics[j]
            
            if self.use_dump:
                output = open(dump_file, 'wb')
                pkl.dump(self.C, output)
        else:
            output = open(dump_file, 'rb')
            self.C = pkl.load(output)
        
        duration = time() - start
        
        # normalizing C
        """
        # worsen performance for z-normalize
        # the l2-norm normalization is applied
        quantile_transformer = preprocessing.QuantileTransformer(
                output_distribution='normal', random_state=0)
        C = quantile_transformer.fit_transform(C)
        """

        C_normalize = preprocessing.normalize(self.C, norm='l2')
        # keep original feature
        # C_normalize = self.C
        
        if self.dimension_reduce:
            # Covariance of C
            Sigma_c = np.cov(C_normalize)
            
            # Run SVD algorithm onto covariance matrix of C
            u, s, vh = np.linalg.svd(Sigma_c, full_matrices=True)
            
            # According to the energy content threshold, select the first k eigenvectors
            totally_variance = sum(s)
            k = len(s)
            for i in range(len(s), 0, -1):
                if sum(s[:i])/totally_variance*100 < self.epsilon:
                    k = i + 1
                    break
            
            # Projecting the features
            C_trans = np.dot(C_normalize.T, u[:, :k])

            # dump the projecting matrix
            self.project_matrix = u[:, :k]
        else:
            C_trans = C_normalize
        
        return C_trans, duration, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters

    def fit(self, X_train, X_test):
        """
        Fix the model and construct the project matrix
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, l_features, m_lengths]
            New data to transform.
            
        Returns
        -------
        None
        """

        X = np.concatenate((X_train, X_test), axis=0)

        C_trans, duration, aggregated_ll_Loss, aggregated_penalty_loss, numberOfParameters = self.fit_transform(X)
        
        C_trans_train = C_trans[:X_train.shape[0]]

        C_trans_test = C_trans[-X_test.shape[0]:]

        return C_trans, C_trans_train, C_trans_test