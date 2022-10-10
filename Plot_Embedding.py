#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:14:30 2020

@author: vito
"""

from sklearn.manifold import TSNE

import numpy as np

import matplotlib.pyplot as plt

def Plot_Embedding(distance, train_length, real_lables, k_means_labels, db_labels, md_db_labels):
    embeddings = TSNE(n_components=2, metric="precomputed", random_state=0).fit_transform(distance)
    
    x_min, x_max = np.min(embeddings, 0), np.max(embeddings, 0)
    data = (embeddings - x_min) / (x_max - x_min)
    
    _font_size = 8
    
    plt.subplot(2, 4, 1)
    
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(real_lables[i])))
    
    plt.xticks([])
    plt.yticks([])
    plt.title("Groud Truth (Full)", fontsize=_font_size)
    
    plt.subplot(2, 4, 2)
    
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(k_means_labels[i])))
    
    plt.xticks([])
    plt.yticks([]) 
    plt.title("K-Means (Full)", fontsize=_font_size)
    
    plt.subplot(2, 4, 3)
    
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(db_labels[i])))
    
    plt.xticks([])
    plt.yticks([])  
    plt.title("DB (Full)", fontsize=_font_size)
    
    plt.subplot(2, 4, 4)
    
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(md_db_labels[i])))
    
    plt.xticks([])
    plt.yticks([])  
    plt.title("MD DB (Full)", fontsize=_font_size)
    
    plt.subplot(2, 4, 5)
    
    for i in range(train_length, data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(real_lables[i])))
    
    plt.xticks([])
    plt.yticks([]) 
    plt.title("Groud Truth (Test)", fontsize=_font_size)
    
    plt.subplot(2, 4, 6)
    
    for i in range(train_length, data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(k_means_labels[i])))
    
    plt.xticks([])
    plt.yticks([]) 
    plt.title("K-Means (Test)", fontsize=_font_size)

    plt.subplot(2, 4, 7)
    
    for i in range(train_length, data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(db_labels[i])))
    
    plt.xticks([])
    plt.yticks([])
    plt.title("DB (Test)", fontsize=_font_size)
    
    plt.subplot(2, 4, 8)
    
    for i in range(train_length, data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='.', c=plt.cm.Set1(int(md_db_labels[i])))
    
    plt.xticks([])
    plt.yticks([]) 
    plt.title("MD DB (Test)", fontsize=_font_size)
    
    fig = plt.gcf()
    fig.set_size_inches(6, 5)
    fig.savefig("Emedding_For_Algos.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    
    
    
    
    
    