#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:12:50 2018

@author: lucien
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split

import time
import numpy as np
import scipy.io as sio
import pr_functions as pr

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T[0:10,:], face_labels.T[0:10,:], test_size=0.2)
X_avg_subs, A_subs, e_vals_subs, e_vecs_subs= pr.compute_eigenspace(X_train, 'low')

for i in range(1, 52):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=0.2)
    X_test = np.append(X_test, X_test_i, axis=0)
    y_test = np.append(y_test, y_test_i, axis=0)
    # Alternative method, compute class subspace, concatenate them horizontally into an array
    X_avg_subs_i, A_subs_i, e_vals_subs_i, e_vecs_subs_i= pr.compute_eigenspace(X_train_i, 'low')
    X_avg_subs = np.vstack((X_avg_subs, X_avg_subs_i))
    A_subs = np.hstack((A_subs, A_subs_i))
    e_vals_subs = np.hstack((e_vals_subs, e_vals_subs_i))
    e_vecs_subs = np.hstack((e_vecs_subs, e_vecs_subs_i))

# Alternative method
test_size = X_test.shape[0]
success_rate_alt=0
mark_alt = 0
E_reconst = np.zeros(52)
for i in range(test_size):
    for j in range(52):
        Wt_subs=np.dot(X_test[i,:]-X_avg_subs[j,:], e_vecs_subs[:,0+8*i:8+8*i])
        Xa=e_vecs_subs[:,0+8*i:8+8*i]*Wt_subs
        X_reconst=X_avg_subs[j,:]+np.sum(Xa, axis=1)
        e_reconst_j=np.linalg.norm(X_reconst-X_test[i,:])
        E_reconst[j] = e_reconst_j
    e_idx_subs = np.argmin(E_reconst)
    if y_train[e_idx_subs] == y_test[i]:
        mark_alt+=1
success_rate_alt = (mark_alt/test_size)*100

print("Success rate using alternative method = %d" % (success_rate_alt))
