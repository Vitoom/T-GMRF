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

    root_path = './Datasets'
        
    select_dataset = dataset_name
    
    print("Processing {0}".format(select_dataset))
    
    dataset_train_path = '{0}/{1}/{1}_TRAIN.arff'.format(root_path, select_dataset)
    dataset_test_path = '{0}/{1}/{1}_TEST.arff'.format(root_path, select_dataset)
    
    X_train, Y_train = read_from_arff(dataset_train_path)
    X_test, Y_test = read_from_arff(dataset_test_path)
    
    return X_train, Y_train, X_test, Y_test

