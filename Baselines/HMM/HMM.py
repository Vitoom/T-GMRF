# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:19:59 2021

@author: wading
"""

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import Normalizer
import math
from scipy.special import kl_div

class HMM(object):
    def __init__(self, n_components=5, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
    
    def fit_transform(self, X):
        
        Prob = np.zeros((X.shape[0], X.shape[0]))
        
        for i in range(X.shape[0]):
    
            model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.n_iter)
            
            model.fit(X[i], lengths=[X[i].shape[0]])
            
            print("model converged: {}".format(model.monitor_.converged))
            
            for j in range(X.shape[0]):
                Prob[i, j] = model.score(X[j], lengths=[X[j].shape[0]])
            
            del model
            
        for i in range(X.shape[0]):
            Prob[i, i] = 1.0
        
        distance = np.zeros((X.shape[0], X.shape[0]))
        
        Z = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            Z[i] = Prob[i, :].sum()
        
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[0]):
                list_i = list(Prob[i])
                list_j = list(Prob[j])
                # distance[i][j] = math.log(sum(abs(Prob[i] - Prob[j])))
                # distance[j][i] = distance[i][j]
                for k in range(X.shape[0]):
                    
                    distance[i][j] += (Prob[i,k]/Z[i]) * math.log((Prob[i,k]/Prob[j,k])*(Z[j]/Z[i])) / 2
                    distance[i][j] += (Prob[j,k]/Z[j]) * math.log((Prob[j,k]/Prob[i,k])*(Z[i]/Z[j])) / 2
                    distance[j][i] = distance[i][j]
                    print("{}".format((Prob[i,k]/Prob[j,k])*(Z[j]/Z[i])))
                    print("{}".format((Prob[j,k]/(Prob[i,k])*(Z[i]/Z[j]))))
                    distance[i][j] += (Prob[i,k]/Z[i]) * math.log((Prob[i,k]/Prob[j,k])*(Z[j]/Z[i])) / 2
                    distance[i][j] += (Prob[j,k]/Z[j]) * math.log((Prob[j,k]/Prob[i,k])*(Z[i]/Z[j])) / 2
                    distance[j][i] = distance[i][j]
                    
        return distance, Prob


if __name__ == '__main__':
    x1 = np.random.normal(50, 100, 1000)
    x2 = np.random.normal(50, 100, 1000)
    x1 = x1.reshape(50, 4, -1)
    x2 = x2.reshape(50, 4, -1)
    x1 = x1.transpose((0, 2, 1))
    x2 = x2.transpose((0, 2, 1))
    
    X = np.concatenate((x1, x2), axis=0)
    Y = [0 for _ in range(50)] + [1 for _ in range(50)]
    
    model = HMM(n_components=5, n_iter=100)
    distance = model.fit_transform(X)