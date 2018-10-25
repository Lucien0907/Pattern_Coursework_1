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

def compute_eigenspace(X_data, mode):
    # Using high/low dimensional computation, return the average vector and the eigenspace of the given data.
    N, D = X_data.shape
    X_avg = X_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    if mode == "high":
        A = (X_data - X_avgm).T    
    elif mode == "low":
        A = (X_data - X_avgm)
    S = A.dot(A.T) / N
    e_vals, e_vecs = np.linalg.eig(S)
    return X_avg, e_vals, e_vecs

def plot_image(face_vector, w, h):
    # Reshape the given image data, plot the image
    image = np.reshape(np.absolute(face_vector),(w,h)).T
    plt.imshow(image, cmap = 'gist_gray')
    #plt.title(title)
    plt.axis('off')
    return

def plot_eig_value(eig_value, i):
    # Plot the first i eigenvalues
    plt.plot(eig_value[:i])
    plt.xlabel('index', fontsize=10)
    plt.ylabel('eigen value', fontsize=10)
    plt.show()
    return

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_labels.T, test_size=0.25, random_state=42)
X_avg, e_vals, e_vecs = compute_eigenspace(X_train, 'high')
"""
plt.subplot(2,1,1)
plot_image(X_avg, 46, 56, 'average face')
"""
# Sort eigen vectors and eigen value in descending order
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T
"""
plt.subplot(2,1,2)
plot_eig_value(e_vals, 30)
"""
# Choose and plot the best M eigenfaces
M = 25
for i in range(M):
    plt.subplot(M/5,5,i+1)
    plot_image(e_vecs[:,i], 46, 56)

# Face reconstruction
