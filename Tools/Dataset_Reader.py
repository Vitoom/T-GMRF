# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 11:00:02 2021

@author: Vito
"""

import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn import preprocessing
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from .HAR_Generate import HAR_Generate
from .DSA_Generate import DSA_Generate
from .EEG_Generate import EEG_Generate

def read_from_arff(path):
    
    f = open(path, 'r', encoding='utf-8')
    data = arff.loadarff(f)
    df = pd.DataFrame(data[0])
    f.close()
    
    X = np.zeros((df[df.columns[0]].shape[0], df[df.columns[0]][0].shape[0], len(df[df.columns[0]][0][0])))
    
    for i in range(df[df.columns[0]].shape[0]):
        for j in range(df[df.columns[0]][0].shape[0]):
            X[i, j, :] = list(df[df.columns[0]][i][j])
    
    trans_label = pd.DataFrame(df[df.columns[1]].astype(str)).applymap(lambda x: x[2:-1])
    
    le = preprocessing.LabelEncoder()
    
    Y = le.fit_transform(trans_label[trans_label.columns[0]])
    
    return X, Y

def ArffDataset_Generate(dataset_name):

    print("Current Directory: \t", os.getcwd()) 

    root_path = './Datasets/Multivariate_arff'
        
    select_dataset = dataset_name
    
    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset_split.pkl"
    if not os.path.exists(dump_file):
        dataset_train_path = '{0}/{1}/{1}_TRAIN.arff'.format(root_path, select_dataset)
        dataset_test_path = '{0}/{1}/{1}_TEST.arff'.format(root_path, select_dataset)
        
        X_train, Y_train = read_from_arff(dataset_train_path)
        X_test, Y_test = read_from_arff(dataset_test_path)

        if select_dataset in ["BasicMotions", "Cricket", "Epilepsy", "JapaneseVowels", "DuckDuckGeese", "EigenWorms", "ERing", "EthanolConcentration", "FingerMovements"]:
            return X_train, Y_train, X_test, Y_test

        X = np.concatenate((X_train,X_test), axis=0)
        Y = np.concatenate((Y_train,Y_test), axis=0)

        # shrink dataset for debug through stratified sampling
        if False:
            _, X, _, Y = train_test_split(X, Y, test_size = 1/10, stratify=Y) # 1/10

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

def Fill_na(X):
    for i in range(X.shape[0]):
        _X = pd.DataFrame(X[i].T).copy()
        if _X.isna().sum().sum() > 0:
            X[i] = _X.fillna(method="ffill").fillna(method="bfill").values.T
    return X

def Get_Dataset(dataset_name):
    print("Starting to get dataset: {0}".format(dataset_name), flush=True)

    dump_file = "./dump/" + dataset_name + "/dataset.pkl"
    directory = os.path.dirname(dump_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    X_train, Y_train, X_test, Y_test = None, None, None, None
    if dataset_name == "HAR":
        X_train, Y_train, X_test, Y_test = HAR_Generate()
    elif dataset_name == "DSA":
        X_train, Y_train, X_test, Y_test = DSA_Generate()
    elif dataset_name == "EEG":
        X_train, Y_train, X_test, Y_test = EEG_Generate()
    else:
        X_train, Y_train, X_test, Y_test = ArffDataset_Generate(dataset_name)

    X_train, X_test = Fill_na(X_train), Fill_na(X_test)

    return X_train, Y_train, X_test, Y_test
