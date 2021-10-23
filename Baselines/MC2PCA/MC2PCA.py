# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:19:59 2021

@author: wading
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import Normalizer

class MC2PCA(object):
    def __init__(self, k=2, tolerance=1e-8, max_iter=300, normalize=False):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.normalize = normalize
        
    def CPCA(self, X, percentile):
        sigma = None
        for i in range(X.shape[0]):
            if i == 0:
                sigma = np.cov(X[i])
            else:
                sigma += np.cov(X[i])
        sigma = sigma / X.shape[1]
        u, s, vh = np.linalg.svd(sigma, full_matrices=True)
        p = 2
        while s[0:p].sum() / s.sum() < percentile:
            p += 1
        S = u[:, :p]
        return S
        
    
    def fit_transform(self, X):
        if self.normalize:
            for i in range(X.shape[1]):
                X[:,i,:] = Normalizer().fit_transform(X[:,i,:])
        
        self.centers_ = {}
        self.assign_ = {}
        self.distance = {}
        
        L = X.shape[0] / self.k_
        
        for i in range(self.k_):
            self.centers_[i] = self.CPCA(X[int(i*L): int(min((i+1)*L, X.shape[0]))], percentile=0.5)
            
        error_sum_pre = 0
        error_sum = 0
        
        n = 0
        
        while 1:
            
            self.assign_ = {}
            
            if n == 0:
                for i in range(self.k_):
                    start = int(i*L)
                    end = int(min((i+1)*L, X.shape[0]))
                    self.assign_[i] = list(range(start, end))
            
            error_sum_pre = error_sum 
            error_sum = 0
            
            pre_assign = self.assign_
            
            remove = []
            
            for i in range(self.k_):
                _recons = lambda index, cluster: np.matmul(np.matmul(X[index].T, self.centers_[cluster]), self.centers_[cluster].T)
                self.distance[i] = [sum(abs(_recons(j, i).reshape(-1) - X[j].reshape(-1))) for j in range(X.shape[0])]
                guarentee = list(np.array(self.distance[i]).argsort()[-min(8, len(self.distance[i])):][::-1])
                self.assign_[i] = guarentee
                error_sum += np.array(self.distance[i])[guarentee].sum()
                remove.append(i)
            
            for i in range(X.shape[0]):
                if i not in remove:
                    recons = lambda cluster: np.matmul(np.matmul(X[i].T, self.centers_[cluster]), self.centers_[cluster].T)
                    select = np.argmin([self.distance[cluster][i] for cluster in range(self.k_)])
                    error_sum += self.distance[select][i]
                
                    if select not in self.assign_.keys():
                        self.assign_[select] = [i,]
                    else:
                        self.assign_[select].append(i)
            
            if abs(error_sum - error_sum_pre) < self.tolerance_:
                print("meet the minimum threshold")
                break
            
            for i in range(self.k_):
                self.centers_[i] = self.CPCA(X[self.assign_[i]], percentile=0.8)
                
            n += 1
            if n >= self.max_iter_:
                print("exceed the max iterations")
                break
        
        label = np.zeros(X.shape[0])
        
        for i in range(self.k_):
            label[self.assign_[i]] = i
        
        return label.astype(int)
    
    def predict(self, p_data): # Don't Normalizing
        recons = lambda cluster: np.matmul(np.matmul(p_data.T, self.centers_[cluster]), self.centers_[cluster].T)
        select = np.argmax([sum(abs(recons(cluster).reshape(-1) - p_data.reshape(-1))) for cluster in range(self.k_)])
        return select


if __name__ == '__main__':
    x1 = np.array(np.random.randint(0, 50, 38400)) / 100
    x2 = np.array(np.random.randint(0, 100, 38400)) / 100
    x1 = x1.reshape(50, 6, -1)
    x2 = x2.reshape(50, 6, -1)
    X = np.concatenate((x1, x2), axis=0)
    Y = [0 for _ in range(50)] + [1 for _ in range(50)]

    k_means = Mc2PCA(k=2)
    label = k_means.fit(X)

    print(adjusted_rand_score(label, Y))
    print(k_means.centers_)