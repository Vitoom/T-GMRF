#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:57:24 2020

@author: vito
"""
import numpy as np
import pandas as pd
import pickle as pkl
import os
from collections import defaultdict
import bisect
from sklearn.model_selection import train_test_split

def DSA_Generate():

    dataset_name = "DSA"

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset.pkl"
    if not os.path.exists(dump_file):
        
        # datastruct to load dataset
        X = np.zeros((19*480, 45, 125))
        Y = np.zeros(19*480)
        dataset = "./Datasets/Daily_and_Sports_Activities/"
        
        # load dataset
        for i in range(1, 20):
            for j in range(1, 9):
                for k in range(1, 61):
                    datafile = dataset + "a{0:0>2d}/p{1}/s{2:0>2d}.txt".format(i, j, k)
                    data = pd.read_csv(datafile, sep=',', header=None)
                    pos = (i-1) * 480 + (j-1) * 60 + k - 1
                    X[pos, :, :] = np.transpose(data.values, (1,0))
                    Y[pos] = i

        directory = os.path.dirname(dump_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        output = open(dump_file, 'wb')
        pkl.dump((X,Y), output)
    else:
        output = open(dump_file, 'rb')
        X, Y = pkl.load(output)
    output.close()
    
    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset_split.pkl"
    if not os.path.exists(dump_file):
        # shrink dataset for debug through stratified sampling
        if True:
            _, X, _, Y = train_test_split(X, Y, test_size = 1/20, stratify=Y) # 1/10

        num_instance = X.shape[0]

        # scaling X
        if False:
            features = X.shape[1]
            lengths = X.shape[2]
            for i in range(num_instance):
                for j in range(features):
                    m = min(X[i, j, :])
                    M = max(X[i, j, :])
                    if M - m != 0:
                        X[i, j, :] = (X[i, j, :] - m) / (M - m)
                    else:
                        X[i, j, :] = X[i, j, :] - m

        X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size = 1/2, stratify=Y)

        output = open(dump_file, 'wb')
        pkl.dump((X_train, X_test, Y_train,  Y_test), output)
    else:
        output = open(dump_file, 'rb')
        X_train, X_test, Y_train,  Y_test = pkl.load(output)
    output.close()
    
    return X_train, Y_train, X_test, Y_test