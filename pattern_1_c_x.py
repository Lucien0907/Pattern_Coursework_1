#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:25:33 2018

@author: lucien
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.io as sio
import pr_functions as pr

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

T = 2
nt = 3
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
        bagx = X_train[index, :]
        bagy = y_train[index, :]
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


# Compute the eigen space of traing data
X_avg, A, e_vals, e_vecs= pr.compute_eigenspace(X_train_bag, 'low')
    
#sort
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T

#randomisation in feature space
M0 = 30
M1 = 30
T_M1 = 10
index_M1 = np.random.choice(range(M0,train_size), M1)
Wpca = np.hstack((e_vecs[:, 0:M0], e_vecs[:, index_M1]))
for t in range(T_M1 - 1):
    index_M1 = np.random.choice(range(M0,train_size), M1)
    Wpca_t = np.hstack((e_vecs[:, 0:M0], e_vecs[:, index_M1]))
    Wpca = np.hstack((Wpca, Wpca_t))

for i in range(T):
    for 
        #calculate Sw, Sb 
        
        X_train_i = bag_x[200*i:200*(i+1), :]
        Mi = np.mean(X_train_i, axis=0)
    
    for j in range(T_M1):



