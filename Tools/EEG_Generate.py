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
from tqdm import tqdm

def EEG_Generate():

    dataset_name = "EEG"

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset.pkl"
    if not os.path.exists(dump_file):
        
        # file path storing the dataset
        path_train = r"./Datasets/SMNI_CMI/SMNI_CMI_TRAIN/"
        folds = []
        for f in os.listdir(path_train):
            if f[0] == 'c':
                folds.append(path_train + f)

        path_test = r"./Datasets/SMNI_CMI/SMNI_CMI_TEST/"
        for f in os.listdir(path_test):
            if f[0] == 'c':
                folds.append(path_test + f)

        file_path_set = []
        for fold in folds:
            files = os.listdir(fold)
            for file in files:
                file_path = fold + "/" + file
                file_path_set.append(file_path)

        # datastruct to load dataset
        X = np.zeros((len(file_path_set), 64, 256))
        Y = np.zeros(len(file_path_set))

        # load dataset
        for i, file in tqdm(enumerate(file_path_set), ascii=True, desc="load EEG dataset"):
            if file.split('/')[-1][3] == 'c':
                Y[i] = 0
            else:
                Y[i] = 1
            f = open(file, mode='r+')
            cur = 0
            data_df = pd.DataFrame(columns=["trail_num", "sensor_pos", "sample_num", "sensor_val"])
            for line in f:
                if line[0] != '#':
                    data_df.loc[data_df.shape[0]] = line[:-1].split(' ')
            for name, group in data_df.groupby(data_df['sensor_pos']):
                X[i, cur, :] = group['sensor_val'].values
                cur += 1

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
            _, X, _, Y = train_test_split(X, Y, test_size = 1/2, stratify=Y) # 1/100

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