
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:19:39 2020

@author: vito
"""

import pickle as pkl
import os
import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import pyreadr
import pandas as pd
from heapq import nlargest

def Align_label(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = pd.Series(y_pred).astype('category').cat.codes
    
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_map = {}
    
    col_max_index = np.argmax(w, axis=1)
    
    max_hits = [w[i][col_max_index[i]] for i in range(len(col_max_index))]
    
    bound = min(nlargest(len(set(y_true)) + 2, max_hits))
    
    for i in range(w.shape[0]):
        if col_max_index[i] >= bound:
            label_map[i] = np.argmax(w[i])
        else:
            label_map[i] = i
    
    y_pred_new = [label_map[ele] for ele in y_pred]

    y_pred = y_pred_new
    
    return y_true, np.array(y_pred)

def Compute_score(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    y_pred = pd.Series(y_pred).astype('category').cat.codes
    
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    col_max_index = np.argmax(w, axis=1)
    
    csm = 0
    
    for i in range(w.shape[0]):
        csm = csm + 1.0 / w.shape[0] * w[i, int(col_max_index[i])] / max(sum(w[i, :]), sum(w[:, int(col_max_index[i])]))
    
    return csm

# computing cluster similarity measure
def cluster_similarity_measure(pred, truth, dataset_name):
    if len(set(truth)) < len(set(pred)):
        truth, pred = Align_label(truth, pred)
    return Compute_score(truth, pred)