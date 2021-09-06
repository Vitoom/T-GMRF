#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:03:57 2019

@author: vito
"""

import numpy as np
import math
import sys
from multiprocessing import Pool
from .admm_solver import ADMMSolver

class TGMRF_solver:
    def __init__(self,
                 width = 10,
                 stride = 1,
                 maxIters = 1000,
                 lr = -1e-3,
                 lamb = 1e-2,
                 beta = 1e-2,
                 threshold = 11e-2,
                 schedule = 'R',
                 epsilon = 1e-6):
        self.width = width
        self.stride = stride
        self.maxIters = maxIters
        self.lr = lr
        self.lamb = lamb
        self.beta = beta
        self.threshold = threshold
        self.schedule = schedule # 'C' for cyclic, 'R' for random
        self.epsilon = epsilon
        self.initilizing = False
        self.num_proc = 4
        
    def upper2Full(self, a):
        n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
        A = np.zeros([n,n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A
    
    def logdet(self, theta):
        # print("a:", np.linalg.det(theta))
        return np.log(np.linalg.det(theta) + self.epsilon)
    
    def Loss(self):
        loss = 0
        shrink = 1 # 1e-5
        for i in range(self.windows_dim):
            theta = self.upper2Full(self.ic_sequence[i]) # self.upper2Full(optRes[i].get())
            if math.isnan(self.logdet(theta)):
                loss += sys.float_info.max * shrink
            else:
                loss += - self.logdet(theta) * shrink
            loss += np.trace(self.upper2Full(self.c_sequence[i, :]) * theta) * shrink + self.beta * abs(theta).sum() * shrink
#            if i == 9:
#                print("block:\t", i, "\t -logdet:", -self.logdet(theta))
            if i > 0 and i < self.windows_dim - 1:
                loss += (self.lamb * np.power(self.upper2Full(self.ic_sequence[i + 1, :]) + self.upper2Full(self.ic_sequence[i - 1, :]) - theta, 2).sum()) * shrink
#        print("loss:", loss)
        return loss

    def LL_Loss(self):
        loss = 0
        shrink = 1 # 1e-5
        for i in range(self.windows_dim):
            theta = self.upper2Full(self.ic_sequence[i]) # self.upper2Full(optRes[i].get())
            if math.isnan(self.logdet(theta)):
                loss += sys.float_info.max * shrink
            else:
                loss += self.logdet(theta) * shrink * 0.5
            loss += - np.trace(self.upper2Full(self.c_sequence[i, :]) * theta) * shrink * 0.5
            loss += - 0.5 * self.variables_dim * math.log(2 * math.pi)
        return loss
    
    def Penalty_Loss(self):
        loss = 0
        shrink = 1 # 1e-5
        for i in range(self.windows_dim):
            theta = self.upper2Full(self.ic_sequence[i])
            loss += - self.beta * abs(theta).sum() * shrink
            if i > 0 and i < self.windows_dim - 1:
                loss += - (self.lamb * np.power(self.upper2Full(self.ic_sequence[i + 1, :]) + self.upper2Full(self.ic_sequence[i - 1, :]) - theta, 2).sum()) * shrink
        return loss
    
    def fit(self, X):
        """
        Main method for ICS solver.
        Parameters:
            - X: the data of multivariate time series
        """
        assert self.maxIters > 0 # must have at least one literation
        
        l_lengths = X.shape[0]
        self.variables_dim = X.shape[1]
        self.windows_dim = int((l_lengths - self.width) / self.stride) + 1
        self.ic_sequence = np.zeros((self.windows_dim, int(self.variables_dim * (self.variables_dim + 1) / 2)))
        self.c_sequence = np.zeros((self.windows_dim, int(self.variables_dim * (self.variables_dim + 1) / 2)))
        
        # initializing Theta with covariance
        for i in range(self.windows_dim):
            _cov = np.cov(X[self.stride * i: self.stride * i + self.width, :].T)
            self.c_sequence[i, :] = _cov[np.triu_indices(self.variables_dim)]
            if self.initilizing:
                self.ic_sequence[i, :] = np.linalg.inv(_cov)[np.triu_indices(self.variables_dim)]
            del _cov
            
        # pool = Pool(processes=self.num_proc)  # multi-threading
        
        # _loss =  0
        if self.schedule == "R":
            k = 0
            while k < self.maxIters:
                # optRes = [None for i in range(self.windows_dim)]
                # ic_sequence_origin = self.ic_sequence.copy()
                for i in range(self.windows_dim):
                    _lamb = np.zeros((self.variables_dim, self.variables_dim)) + self.lamb
                    _beta = np.zeros((self.variables_dim, self.variables_dim)) + self.beta
                    solver = ADMMSolver(self.ic_sequence, i, _lamb, _beta, self.variables_dim, self.windows_dim, 1,
                                        self.upper2Full(self.c_sequence[i]))
                    self.ic_sequence[i] = solver.__call__(1000, 1e-6, 1e-6, False) # 1000
                    # optRes[i] = solver.__call__(1000, 1e-6, 1e-6, False) # pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
                    # self.ic_sequence[i] = optRes[i]
                    # print("turn:\t ", k, "\t error for para:\t",np.linalg.norm(self.ic_sequence - ic_sequence_origin, ord = 2))
                    k += 1
                    if k >= self.maxIters:
                        break
                loss = self.Loss()
                ll_loss = self.LL_Loss()
                penalty_loss = self.Penalty_Loss()
                numberOfParameters = self.windows_dim * int(self.variables_dim * (self.variables_dim + 1) / 2)
                # print("turn:\t ", k, "\t loss:{}({})\t".format(loss, loss-_loss))
                # _loss = self.Loss()
        return self.ic_sequence, loss, ll_loss, penalty_loss, numberOfParameters

if __name__ == '__main__':
    a = np.random.rand(100, 10)
    clf = TGMRF_solver(width=10, stride=10, maxIters=1000, lr=0, lamb=1e-2, beta=1e-2)
    b = clf.fit(a)
