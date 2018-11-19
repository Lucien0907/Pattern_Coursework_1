#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:12:50 2018

@author: lucien
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

for i in range(1, 52):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=0.2)
    X_train = np.append(X_train, X_train_i, axis=0)
    X_test = np.append(X_test, X_test_i, axis=0)
    y_train = np.append(y_train, y_train_i, axis=0)
    y_test = np.append(y_test, y_test_i, axis=0)

# Compute the eigen space of traing data
start_time = time.time()
X_avg, A, e_vals, e_vecs= pr.compute_eigenspace(X_train, 'high')
t_1 = time.time()
X_avg, A, e_vals, e_vecs= pr.compute_eigenspace(X_train, 'low')
t_2 = time.time()
print(t_1-start_time)
print(t_2 - t_1)

# Sort eigen vectors and eigen value in descending order
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T

m=100
test_size = X_test.shape[0]
mark = 0
for i in range(test_size):
        Wt=np.dot(X_test[i,:]-X_avg, e_vecs[:,:m])
        Wn=np.dot(A.T, e_vecs[:,:m])
        E = np.linalg.norm(Wt.T - Wn, axis=1)
        e_idx = np.argmin(E)
        if i == 0:
            y_pred = y_train[e_idx]
        else:
            y_pred = np.hstack((y_pred, y_train[e_idx]))
        if y_train[e_idx] == y_test[i]:
            test_success = i
            pred_success = e_idx
            mark+=1
        else:
            test_fail = i
            pred_fail = e_idx

pr.plot_image(X_test[test_success,:], 46, 56, 'Outputs/nn_test_success')
pr.plot_image(X_train[pred_success,:], 46, 56, 'Outputs/nn_pred_success')
X_proj=np.dot(X_test[test_success,:]-X_avg, e_vecs[:,:m])
Xa=e_vecs[:,:m]*X_proj
X_reconst=X_avg+np.sum(Xa, axis=1)
pr.plot_image(X_reconst, 46, 56, 'Outputs/nn_reconst_success')

pr.plot_image(X_test[test_fail,:], 46, 56, 'Outputs/nn_test_fail')
pr.plot_image(X_train[pred_fail,:], 46, 56, 'Outputs/nn_pred_fail')
X_proj=np.dot(X_test[test_fail,:]-X_avg, e_vecs[:,:m])
Xa=e_vecs[:,:m]*X_proj
X_reconst=X_avg+np.sum(Xa, axis=1)
pr.plot_image(X_reconst, 46, 56, 'Outputs/nn_reconst_fail')
    
# Compute confusion matrix
success_rate = (mark/test_size)*100
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cnf_matrix, cmap="Blues", xticklabels = 5, yticklabels = 5)

sns.plt.show()
print(success_rate)

"""
t_3=time.time()

# Classification using NN
M_range=415
test_size = X_test.shape[0]
correctness = np.zeros(M_range)
for m in range(M_range):
    # This loop could take a while depends on size of M_range
    mark = 0
    for i in range(test_size):
        Wt=np.dot(X_test[i,:]-X_avg, e_vecs[:,:m])
        Wn=np.dot(A.T, e_vecs[:,:m])
        E = np.linalg.norm(Wt.T - Wn, axis=1)
        e_idx = np.argmin(E)
        if y_train[e_idx] == y_test[i]:
            mark+=1
    correctness[m] = (mark/test_size)*100
t_4=time.time()
# Plot success rate against M
pr.plot_graph("line", correctness, M_range, 'M', 'success rate / %', 50, 5, 'Outputs/success_rate_against_M')
print("Highest succes rate %.2f%% when M = %d" % (np.max(correctness), np.argmax(correctness)))
print(t_4)
"""