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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_eigenspace(X_data, mode):
    # Using high/low dimensional computation, return the average vector and the eigenspace of the given data.
    N, D = X_data.shape
    X_avg = X_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    A = (X_data - X_avgm).T
    if mode == "high":
        S = A.dot(A.T) / N
        e_vals, e_vecs = np.linalg.eig(S)
    elif mode == "low":
        S = (A.T).dot(A) / N
        e_vals, e_vecs = np.linalg.eig(S)
        e_vecs = np.dot(A,e_vecs)
        e_vecs = e_vecs / np.linalg.norm(e_vecs, axis=0)
    return X_avg, A, e_vals, e_vecs

def plot_image(face_vector, w, h, filename):
    # Reshape the given image data, plot the image
    plt.figure()
    image = np.reshape(np.absolute(face_vector),(w,h)).T
    fig = plt.imshow(image, cmap = 'gist_gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    plt.close()
    return

def plot_graph(type, eig_value, i, x, y, xtick, ytick, filename):
    # Plot the first i eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if type == "bar":
        plt.bar(list(range(0, i)), eig_value[:i])
    else:plt.plot(list(range(0, i)), eig_value[:i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    #plt.show(block=False)
    return

#Load image data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_labels.T, test_size=0.3, random_state=42)
X_avg, A, e_vals, e_vecs= compute_eigenspace(X_train, 'low')

# Plot mean face
plot_image(X_avg, 46, 56, 'Outputs/mean_face')

# Sort eigen vectors and eigen value in descending order
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T

# Choose and plot the best M eigenfaces
M = 20
plot_graph("bar", e_vals, M, 'index', 'eigen_value', 1, 100000, 'Outputs/first_'+str(M)+'_eigenvalues')
for i in range(M):
    plot_image(e_vecs[:,i], 46, 56, 'Outputs/Eigenfaces/eigenface_'+str(i))

# Face reconstruction
test1=29
X_proj=np.dot(A[:,test1].T, e_vecs[:,:M])
Xa=e_vecs[:,:M]*X_proj
X_reconst=X_avg+np.sum(Xa, axis=1)

# Plot the chosen test face & reconstructed face for comparison
plot_image(X_test[test1,:], 46, 56, 'Outputs/test_face_M='+str(M))
plot_image(X_reconst, 46, 56, 'Outputs/reconstructed_face_M='+str(M))

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
plot_graph("line", correctness, M_range, 'M', 'success rate', 5, 5, 'Outputs/success_rate_against_M')
print("Highest succes rate %.2f%% when M = %d" % (np.max(correctness), np.argmax(correctness)))
#plt.show()
