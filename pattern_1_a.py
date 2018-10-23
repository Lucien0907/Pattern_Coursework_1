#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:42:55 2018

@author: lucien
"""

from __future__ import print_function

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_labels.T, test_size=0.25, random_state=42)

N, D = X_train.shape
# Find and print out the mean face
X_avg = X_train.mean(0)
face_mean = np.reshape(X_avg,(46, 56)).T
plt.subplot(2,1,1)
plt.imshow(face_mean, cmap = 'gist_gray')
plt.title('mean face')

# Compute eigen vecters and eigen values
X_avgm = np.array([X_avg]*N)
A = (X_train - X_avgm).T
S = A.dot(A.T) / N
e_vals, e_vecs = np.linalg.eig(S)

# Plot eigen faces
face_eig = np.reshape(np.absolute(e_vecs[:,3]),(46,56)).T
plt.subplot(2,1,2)
plt.imshow(face_eig, cmap = 'gist_gray')
plt.title('eigen face sample')

# Print the graph of eigen values against eigen vectors
# Keep the best M eigenvectors