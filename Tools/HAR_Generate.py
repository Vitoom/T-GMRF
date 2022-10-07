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

def HAR_Generate():

    dataset_name = "HAR"

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset_train.pkl"
    if not os.path.exists(dump_file):
        
        # datastruct to load dataset
        X_train = np.zeros((7352, 6, 128))
        # Y = np.zeros(7352)
        
        # load dataset
        dataset_x = r"./Datasets/UCI HAR Dataset/train/Inertial Signals/"
        dataset_y = r"./Datasets/UCI HAR Dataset/train/y_train.txt"
        datafiles = [r"body_acc_x_train.txt", r"body_acc_y_train.txt", r"body_acc_z_train.txt",
                    r"body_gyro_x_train.txt", r"body_gyro_y_train.txt", r"body_gyro_z_train.txt"]
        for i in range(6):
            # datafile = dataset_x + datafiles[i]
            data = pd.read_csv(dataset_x + datafiles[i], sep='\s+', header=None)
            X_train[:, i, :] = data.values
        data = pd.read_csv(dataset_y, sep='\s+', header=None)
        Y_train = data[0].values

        directory = os.path.dirname(dump_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        output = open(dump_file, 'wb')
        pkl.dump((X_train,Y_train), output)
    else:
        output = open(dump_file, 'rb')
        X_train, Y_train = pkl.load(output)
    output.close()

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset_test.pkl"
    if not os.path.exists(dump_file):
        
        # datastruct to load dataset
        X_test = np.zeros((2947, 6, 128))
        # Y = np.zeros(7352)
        
        # load dataset
        dataset_x = r"./Datasets/UCI HAR Dataset/test/Inertial Signals/"
        dataset_y = r"./Datasets/UCI HAR Dataset/test/y_test.txt"
        datafiles = [r"body_acc_x_test.txt", r"body_acc_y_test.txt", r"body_acc_z_test.txt",
                    r"body_gyro_x_test.txt", r"body_gyro_y_test.txt", r"body_gyro_z_test.txt"]
        for i in range(6):
            # datafile = dataset_x + datafiles[i]
            data = pd.read_csv(dataset_x + datafiles[i], sep='\s+', header=None)
            X_test[:, i, :] = data.values
        data = pd.read_csv(dataset_y, sep='\s+', header=None)
        Y_test = data[0].values

        directory = os.path.dirname(dump_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        output = open(dump_file, 'wb')
        pkl.dump((X_test,Y_test), output)
    else:
        output = open(dump_file, 'rb')
        X_test, Y_test = pkl.load(output)
    output.close()

    X = np.concatenate((X_train,X_test), axis=0)
    Y = np.concatenate((Y_train,Y_test), axis=0)

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset_split.pkl"
    if not os.path.exists(dump_file):

        # shrink dataset for debug through stratified sampling
        if True:
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