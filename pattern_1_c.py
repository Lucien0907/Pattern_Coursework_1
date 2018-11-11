#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:41:01 2018

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

X_train, X_test, y_train, y_test = train_test_split(face_data.T[0:10,:], face_labels.T[0:10,:], test_size=0.2)
Mi = np.mean(X_train, axis=0)
Si = np.zeros((2576,2576))
for v in X_train:
    S0 = np.outer((v - Mi), (v - Mi))
    Si = Si + S0
Sw = Si

# Split into a training and testing set
for i in range(1, 52):
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(face_data.T[0+i*10:10+i*10,:], face_labels.T[0+i*10:10+i*10,:], test_size=0.2)
    m = np.mean(X_train_rand, axis=0)
    for v in X_train_rand:
        S0 = np.outer((v - m), (v - m))
        Si = Si + S0
    Sw = Sw + Si
    Mi = np.vstack((Mi, m))
    X_train = np.append(X_train, X_train_rand, axis=0)
    X_test = np.append(X_test, X_test_rand, axis=0)
    y_train = np.append(y_train, y_train_rand, axis=0)
    y_test = np.append(y_test, y_test_rand, axis=0)

M = np.mean(Mi, axis=0)
Sb = np.dot((Mi - M).T,(Mi - M))

A = (X_train - M)
St = A.dot(A.T)
e_vals_pca, e_vecs = np.linalg.eig(St)
e_vecs = np.dot(A.T,e_vecs)
e_vecs_pca = e_vecs / np.linalg.norm(e_vecs, axis=0)

idx2=np.argsort(np.absolute(e_vals_pca))[::-1]
e_vals_pca = e_vals_pca[idx2]
e_vecs_pca = (e_vecs_pca.T[idx2]).T
Mpca = 100
Wpca = e_vecs_pca[:,:Mpca]

SBB = np.dot(Wpca.T, Sb)
SBB = np.dot(SBB, Wpca)
SWW = np.dot(Wpca.T, Sw)
SWW = np.dot(SWW, Wpca)
e_vals_lda, e_vecs_lda = np.linalg.eig(np.dot(np.linalg.inv(SWW), SBB))

idx1=np.argsort(np.absolute(e_vals_lda))[::-1]
e_vals_lda = e_vals_lda[idx1]
e_vecs_lda = (e_vecs_lda.T[idx1]).T
Mlda = 50
Wlda = e_vecs_lda[:,:Mlda]

Wopt_t = np.dot(Wlda.T, Wpca.T)

#plot_graph("bar", Wopt_t, Mlda, 'index', 'eigen_value', 1, 100000, 'Outputs_c/eigenvlues')
for i in range(Mlda):
    plot_image(Wopt_t.T[:,i], 46, 56, 'Outputs_c/fisher_faces_'+str(i))

test_size = X_test.shape[0]
correctness = np.zeros(Mlda)
for m in range(Mlda):
    # This loop could take a while depends on size of M_range
    mark = 0
    for i in range(test_size):
       W_train = np.dot(X_train - M, Wopt_t.T)
       W_test = np.dot(X_test[i,:] - M, Wopt_t.T)
       E = np.linalg.norm(W_test - W_train, axis=1)
       e_idx = np.argmin(E)
       if y_train[e_idx] == y_test[i]:
           mark+=1
    correctness[m] = (mark/test_size)*100
    
plot_graph("line", correctness, Mlda, 'M', 'success rate', 5, 5, 'Outputs_c/success_rate_against_M')
print("Highest succes rate %.2f%% when M = %d" % (np.max(correctness), np.argmax(correctness)))

test = 15
W_test = np.dot(X_test[test,:] - M, Wopt_t.T)
Xa=Wopt_t[:Mlda,:].T*W_test
X_reconst=M+np.sum(Xa, axis=1)

plot_image(X_test[test,:], 46, 56, 'Outputs_c/test_face_M='+str(Mlda))
plot_image(X_reconst, 46, 56, 'Outputs_c/reconstructed_face_M='+str(Mlda))