#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:29:46 2018

@author: panzengyang
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

T = 3
nt = 6
for t in range(T):
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(face_data.T[0:10,:], face_labels.T[0:10,:], test_size=0.2)
    train_size = X_train.shape[0]
    #bagging
    index = np.random.choice(train_size, nt)
    X_train_t = X_train[index, :]
    y_train_t = y_train[index, :]
    Mi = np.mean(X_train_t, axis=0)
    Sw = np.zeros((2576,2576))
    for v in X_train_t:
        S0 = np.outer((v - Mi), (v - Mi))
        Sw = Sw + S0
    for i in range(1, 52):
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=0.2)
        train_size = X_train_rand.shape[0]
        index = np.random.choice(train_size, nt)
        bagx = X_train_rand[index, :]
        bagy = y_train_rand[index, :]
        mi = np.mean(bagx, axis=0)
        for v in bagx:
            S0 = np.outer((v - mi), (v - mi))
            Sw = Sw + S0
        Mi = np.vstack((Mi, mi))
        X_train_t = np.append(X_train_t, bagx, axis=0)
        X_test = np.append(X_test, X_test_rand, axis=0)
        y_train_t = np.append(y_train_t, bagy, axis=0)
        y_test = np.append(y_test, y_test_rand, axis=0)
    if(t == 0):
        Mi_bag = Mi
        Sw_bag = Sw
        X_train_bag = X_train_t
        y_train_bag = y_train_t
    else:
        Mi_bag = np.hstack((Mi_bag, Mi))
        Sw_bag = np.hstack((Sw_bag, Sw))
        X_train_bag = np.hstack((X_train_bag, X_train_t))
        y_train_bag = np.hstack((y_train_bag, y_train_t))


train_size = X_train_bag.shape[0]
for i in range(T):
    # Compute Sb,Sw and mean face
    Mt = Mi_bag[:,i*2576:(i+1)*2576]
    M = np.mean(Mt, axis=0)
    Sb = np.dot((Mt - M).T,(Mt - M))
    #overwrite precious Sw, Sw for sepcific bag
    Sw = Sw_bag[:,i*2576:(i+1)*2576]
    # Compute eigenface for feature space
    #overwrite precious X_train, X_train for sepcific bag
    X_train = X_train_bag[:,i*2576:(i+1)*2576]
    A = (X_train - M)
    St = A.dot(A.T)
    e_vals_pca, e_vecs = np.linalg.eig(St)
    e_vecs = np.dot(A.T,e_vecs)
    e_vecs_pca = e_vecs / np.linalg.norm(e_vecs, axis=0)
    
    # Sort and pick the best Mpca eigenvectors
    idx_pca=np.argsort(np.absolute(e_vals_pca))[::-1]
    e_vals_pca = e_vals_pca[idx_pca]
    e_vecs_pca = (e_vecs_pca.T[idx_pca]).T
    #randomisation in feature space
    M0 = 30
    M1 = 30
    T_M1 = 3
    Mlda = 25
    for t in range(T_M1):
        #random select M1 without replacement
        index_M1 = np.random.choice(range(M0,train_size), M1, replace=False)
        Wpca = np.hstack((e_vecs_pca[:, 0:M0], e_vecs_pca[:, index_M1]))
        # Compute eigen space Wlda
        SB = np.dot(np.dot(Wpca.T, Sb), Wpca)
        SW = np.dot(np.dot(Wpca.T, Sw), Wpca)
        e_vals_lda, e_vecs_lda = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))
        # Sort and choose the best Mlda eigenvectors
        idx1=np.argsort(np.absolute(e_vals_lda))[::-1]
        e_vals_lda = e_vals_lda[idx1]
        e_vecs_lda = (e_vecs_lda.T[idx1]).T
        Wlda = e_vecs_lda[:,:Mlda]
        # Optimal fisherspace
        if(t == 0):
            Wopt_t = np.dot(Wlda.T, Wpca.T)
        else:
            Wopt_t = np.vstack((Wopt_t, np.dot(Wlda.T, Wpca.T)))
            
    if(i == 0):
        W_OPT = Wopt_t
    else:
        W_OPT = np.hstack((W_OPT, Wopt_t))


#correctness for individual bags
test_size = X_test.shape[0]
correctness = np.zeros((T_M1,T))    
for j in range(T):
    Mt = Mi_bag[:,j*2576:(j+1)*2576]
    M = np.mean(Mt, axis=0)
    X_train = X_train_bag[:,j*2576:(j+1)*2576]
    y_train = y_train_bag[:,j:(j+1)]
    Wopt_t = W_OPT[:,j*2576:(j+1)*2576]
    for k in range(T_M1):
        Wopt = Wopt_t[k*Mlda:(k+1)*Mlda,:]
        mark = 0
        for i in range(test_size):
            W_train = np.dot(X_train - M, Wopt.T)
            W_test = np.dot(X_test[i,:] - M, Wopt.T)
            E = np.linalg.norm(W_test - W_train, axis=1)
            e_idx = np.argmin(E)
            if y_train[e_idx] == y_test[i]:
                mark+=1
        correctness[k,j] = (mark/test_size)*100



test_size = X_test.shape[0]
mark = 0
for i in range(test_size):   
    for j in range(T):
        Mt = Mi_bag[:,j*2576:(j+1)*2576]
        M = np.mean(Mt, axis=0)
        X_train = X_train_bag[:,j*2576:(j+1)*2576]
        y_train = y_train_bag[:,j:(j+1)]
        Wopt_t = W_OPT[:,j*2576:(j+1)*2576]
        for k in range(T_M1):
            Wopt = Wopt_t[k*Mlda:(k+1)*Mlda,:]
            W_train = np.dot(X_train - M, Wopt.T)
            W_test = np.dot(X_test[i,:] - M, Wopt.T)
            E = np.linalg.norm(W_test - W_train, axis=1)
            e_idx_k = np.argmin(E)
            idx_k = y_train[e_idx_k]
            if(k == 0):
                idx = idx_k
            else:
                idx = np.hstack((idx, idx_k))
        if(j == 0):
            IDX = idx
        else:
            IDX = np.hstack((IDX, idx))                          
    idx_m = np.bincount(IDX).argmax()
    if idx_m == y_test[i]:
        mark+=1
    if(i == 0):
        y_pred = idx_m
    else:
        y_pred = np.hstack((y_pred, idx_m))
        
crtness = (mark/test_size)*100
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(cnf_matrix, cmap='hot', interpolation='nearest')
plt.show()