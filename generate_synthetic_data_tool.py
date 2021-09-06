# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 23:32:32 2021

@author: Vito
"""

import numpy as np
from snap import *
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def generate_inverse(rand_seed, num_blocks, size_blocks, sparsity_inv_matrix, block_matrices):
    np.random.seed(rand_seed)
    def genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2, symmetric = True):
        portion = portion/2
        S = np.zeros((size,size))
        G = GenRndGnm(PNGraph, size, int((size*(size-1))*portion))
        for EI in G.Edges():
            value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
            S[EI.GetSrcNId(), EI.GetDstNId()] = value
        if symmetric:
            S = S + S.T
        return np.matrix(S)

    def genRandInv(size,low = 0.3, upper=0.6, portion = 0.2):
        S = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                if i==j: #and np.random.rand() < portion:
                    # value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0]) 
                    S[i,j] = 1
        return np.matrix(S)

    ##Generate all the blocks
    for block in range(num_blocks):
        if block ==0:
            block_matrices[block] = genInvCov(size = size_blocks, portion = sparsity_inv_matrix, symmetric = (block == 0) )
        else:
            block_matrices[block] = genRandInv(size = size_blocks, portion = sparsity_inv_matrix)

    ##Initialize the inverse matrix
    inv_matrix = np.zeros([num_blocks*size_blocks,num_blocks*size_blocks])

    ##go through all the blocks
    for block_i in range(num_blocks):
        for block_j in range(num_blocks):
            block_num = np.abs(block_i - block_j)
            if block_i > block_j:
                inv_matrix[block_i*size_blocks:(block_i+1)*size_blocks, block_j*size_blocks:(block_j+1)*size_blocks] = block_matrices[block_num]
            else:
                inv_matrix[block_i*size_blocks:(block_i+1)*size_blocks, block_j*size_blocks:(block_j+1)*size_blocks] = np.transpose(block_matrices[block_num])

    ##print out all the eigenvalues
    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)

    ##Make the matrix positive definite
    inv_matrix = inv_matrix + (0.1 + abs(lambda_min))*np.identity(size_blocks*num_blocks)

    eigs, _ = np.linalg.eig(inv_matrix)
    lambda_min = min(eigs)
    print("Modified Eigenvalues are:", np.sort(eigs))

    ##Save the matrix to file
    # np.savetxt("matrix_random_seed=" + str(rand_seed) + ".csv", inv_matrix, delimiter =",",fmt='%1.2f')
    return inv_matrix


def Generating_Fake_Dataset(_Ts_length, _data_size, _dimension):

    Ts_length = _Ts_length
    data_size = _data_size
    dimension = _dimension  
    
    window_size = 5
    number_of_sensors = dimension
    sparsity_inv_matrix = 0.2
    rand_seed = 10
    number_of_clusters = 3
    cluster_ids = [0,1,2]
    break_points = np.array([1,2,3]) * int(Ts_length / 3)
    save_inverse_covarainces = False
    
    size_blocks = number_of_sensors
    
    Dataset = np.zeros([data_size, break_points[-1], size_blocks])
    
    for i in range(3): # 3 MTS clusters
    
        print("Generating {} cluster".format(i + 1))
    
        block_matrices = {} ##Stores all the block matrices
        num_blocks = window_size
        
        sparsity_inv_matrix = sparsity_inv_matrix
        block_matrices = {} ##Stores all the block matrices
        seg_ids = cluster_ids
        
        ############GENERATE POINTS
        num_clusters = number_of_clusters
        cluster_mean = np.zeros([size_blocks,1])
        cluster_mean_stacked = np.zeros([size_blocks*num_blocks,1])
        
        ##Generate two inverse matrices
        cluster_inverses = {}
        cluster_covariances = {}
        for cluster in range(num_clusters):
            cluster_inverses[cluster] = generate_inverse(rand_seed = cluster, 
                                                      num_blocks=num_blocks,
                                                      size_blocks=size_blocks, 
                                                      sparsity_inv_matrix=sparsity_inv_matrix,
                                                      block_matrices=block_matrices)
            cluster_covariances[cluster] = np.linalg.inv(cluster_inverses[cluster])
            if save_inverse_covarainces:
                np.savetxt("Inverse Covariance cluster ="+ str(cluster) +".csv", cluster_inverses[cluster],delimiter= ",",fmt='%1.6f')
                np.savetxt("Covariance cluster ="+ str(cluster) +".csv", cluster_covariances[cluster],delimiter= ",",fmt='%1.6f')
        
        for j in range(int(data_size/3)):
            ##Data matrix
            Data = np.zeros([break_points[-1],size_blocks])
            Data_stacked = np.zeros([break_points[-1]-num_blocks+1, size_blocks*num_blocks])
            cluster_point_list = []
            for counter in range(len(break_points)):
                break_pt = break_points[counter]
                cluster = seg_ids[counter]
                if counter == 0:
                    old_break_pt = 0
                else:
                    old_break_pt = break_points[counter-1]
                for num in range(old_break_pt,break_pt):
                    ##generate the point from this cluster
                    # print("num is:", num)
                    if num == 0:
                        cov_matrix = cluster_covariances[cluster][0:size_blocks,0:size_blocks]##the actual covariance matrix
                        new_mean = cluster_mean_stacked[size_blocks*(num_blocks-1):size_blocks*num_blocks]
                        ##Generate data            
                        # print("new mean is:", new_mean)
                        # print("size_blocks:", size_blocks)
                        # print("cov_matrix is:", cov_matrix)
                        new_row = np.random.multivariate_normal(new_mean.reshape(size_blocks),cov_matrix)
                        Data[num,:] = new_row
            
                    elif num < num_blocks:
                        ##The first section
                        cov_matrix = cluster_covariances[cluster][0:(num+1)*size_blocks,0:(num+1)*size_blocks] ##the actual covariance matrix
                        n = size_blocks
                        Sig22 = cov_matrix[(num)*n:(num+1)*n,(num)*n:(num+1)*n] 
                        Sig11 = cov_matrix[0:(num)*n,0:(num)*n]
                        Sig21 = cov_matrix[(num)*n:(num+1)*n,0:(num)*n]
                        Sig12 = np.transpose(Sig21)
                        cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
                        log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
                        inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1
            
                        ##Generate data
                        a = np.zeros([(num)*size_blocks,1])
                        for idx in range(num):
                            a[idx*size_blocks:(idx+1)*size_blocks,0] = Data[idx,: ].reshape([size_blocks])
                        new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),(a - cluster_mean_stacked[0:(num)*size_blocks,:]) )
                        new_row = np.random.multivariate_normal(new_mean.reshape(size_blocks),cov_mat_tom)
                        Data[num,:] = new_row
            
                    else:
                        cov_matrix = cluster_covariances[cluster]##the actual covariance matrix
                        n = size_blocks
                        Sig22 = cov_matrix[(num_blocks-1)*n:(num_blocks)*n,(num_blocks-1)*n:(num_blocks)*n] 
                        Sig11 = cov_matrix[0:(num_blocks-1)*n,0:(num_blocks-1)*n]
                        Sig21 = cov_matrix[(num_blocks-1)*n:(num_blocks)*n,0:(num_blocks-1)*n]
                        Sig12 = np.transpose(Sig21)
                        cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
                        log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
                        inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1
            
                        ##Generate data
                        # print("shape of the inv_cov_mat_tom is:", inv_cov_mat_tom.shape)
                        # print("cov_mat_tom", cov_mat_tom.shape)
                        # print("Sig11 shape", Sig11.shape)
            
                        a = np.zeros([(num_blocks-1)*size_blocks,1])
                        for idx in range(num_blocks-1):
                            a[idx*size_blocks:(idx+1)*size_blocks,0] = Data[num - num_blocks + 1 + idx,: ].reshape([size_blocks])
                        # print("shape cluster_mean stacked is:", (cluster_mean_stacked[0:(num)*size_blocks,:]).shape)
                        # print("shape of a is", (a).shape# - cluster_mean_stacked[0:(num)*size_blocks,0]).shape)
            
                        new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),(a - cluster_mean_stacked[0:(num_blocks-1)*size_blocks,:]) )
                        # print("shape of new_mean is:", new_mean.shape)
                        # print("new mean is:", new_mean)
                        # print("size_blocks:", size_blocks)
            
                        new_row = np.random.multivariate_normal(new_mean.reshape(size_blocks),cov_mat_tom)
                        Data[num,:] = new_row
            
            Dataset[i * int(data_size / 3) + j, :, :] = Data
    
    print("done with generating the data!!!")
            
    return Dataset

def Fake_Dataset(dataset_name):

    l = dataset_name.split("_")

    # save or load datastruct for time-saving --- dataset
    dump_file = "./dump/" + dataset_name + "/dataset.pkl"
    if not os.path.exists(dump_file):
        
        X = Generating_Fake_Dataset(int(l[1]), int(l[2]), int(l[3]))

        instancePerCluster = int(int(l[2]) / 3)

        Y = []
        for i in range(3):
            Y = Y + [i] * instancePerCluster

        directory = os.path.dirname(dump_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        output = open(dump_file, 'wb')
        pkl.dump((X,Y), output)
    else:
        output = open(dump_file, 'rb')
        X, Y = pkl.load(output)
    output.close()

    return X, Y

if __name__ == '__main__':

    Ts_length = 3 * 100
    data_size = 3 * (10 ** 2)
    dimension = 4

    dataset = Generating_Fake_Dataset(Ts_length, data_size, dimension)