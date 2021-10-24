
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

# computing cluster similarity measure
def cluster_similarity_measure(pred, truth, dataset_name):
    dataset_file_1 = "./" + dataset_name + "/cluster_result_1" + ".Rds"
    directory = os.path.dirname(dataset_file_1)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pyreadr.write_rds(dataset_file_1, pd.DataFrame(pred))
    dataset_file_2 = "./" + dataset_name + "/cluster_result_2" + ".Rds"
    pyreadr.write_rds(dataset_file_2, pd.DataFrame(truth))
    command = 'sh ./measure.sh ' + os.getcwd() + " " + dataset_name
    ret = os.system(command)
    os.remove(dataset_file_1)
    os.remove(dataset_file_2)
    if ret != 0:
        raise ValueError("Error: measure.R script runing error!")
    else:
        with open("./measure.txt", "r") as f:
            csm = f.readline().split(' ')[0]
    return csm