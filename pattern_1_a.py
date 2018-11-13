#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:42:55 2018
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

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_labels.T, test_size=0.3, random_state=42)
X_avg, A, e_vals, e_vecs= pr.compute_eigenspace(X_train, 'low')

# Plot mean face
pr.plot_image(X_avg, 46, 56, 'Outputs/mean_face')

# Sort eigen vectors and eigen value in descending order
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T

# Choose and plot the best M eigenfaces
M = 20
pr.plot_graph("bar", e_vals, M, 'index', 'eigen_value', 1, 100000, 'Outputs/first_'+str(M)+'_eigenvalues')
for i in range(M):
    pr.plot_image(e_vecs[:,i], 46, 56, 'Outputs/Eigenfaces/eigenface_'+str(i))

# Face reconstruction
test1=29
X_proj=np.dot(A[:,test1].T, e_vecs[:,:M])
Xa=e_vecs[:,:M]*X_proj
X_reconst=X_avg+np.sum(Xa, axis=1)

# Plot the chosen test face & reconstructed face for comparison
pr.plot_image(X_test[test1,:], 46, 56, 'Outputs/test_face_M='+str(M))
pr.plot_image(X_reconst, 46, 56, 'Outputs/reconstructed_face_M='+str(M))

# Classification
M_range=30
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

# Plot success rate against M
pr.plot_graph("line", correctness, M_range, 'M', 'success rate', 5, 5, 'Outputs/success_rate_against_M')
print("Highest succes rate %.2f%% when M = %d" % (np.max(correctness), np.argmax(correctness)))
#plt.show()
