#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import math
from decimal import Decimal


# In[51]:


def vector_norm(x):
    """Compute the L2 norm of a vector."""
    return Decimal(np.dot(x, x)).sqrt()


def qr_decomposition(A):
    """
    Compute the QR decomposition of a matrix A using the Gram-Schmidt algorithm.

    Parameters:
    A (numpy.ndarray): The matrix to decompose.

    Returns:
    (numpy.ndarray, numpy.ndarray): A tuple of Q and R matrices that represent the QR decomposition of A, where:
        Q (numpy.ndarray): The orthogonal matrix Q.
        R (numpy.ndarray): The upper triangular matrix R.
    """
    # Get the shape of the input matrix
    m, n = A.shape

    # Initialize the matrices
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Perform the Gram-Schmidt orthogonalization
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        # R[j, j] = np.linalg.norm(v)
        R[j, j] = vector_norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


# In[59]:


def eig(A):
    """
    Compute the eigenvalues and eigenvectors of a matrix A using the power iteration method.

    Parameters:
    A (numpy.ndarray): The matrix to compute eigenvalues and eigenvectors.

    Returns:
    (eigvals, eigvecs): A tuple of arrays that represent the eigenvalues and eigenvectors of A, where:
        eigvals (numpy.ndarray): The eigenvalues of A.
        eigvecs (numpy.ndarray): The eigenvectors of A.
    """
    # set the number of iterations and tolerance level
    max_iter = 100
    tol = 1e-6

    # initialize the eigenvectors
    m, n = A.shape
    eigvecs = np.random.randn(n, n)

    # compute the largest eigenvalue and eigenvector
    for i in range(max_iter):
        # compute the new eigenvector
        eigvecs_new = A @ eigvecs
        # eigvecs_new, _ = np.linalg.qr(eigvecs_new)
        eigvecs_new, _ = qr_decomposition(eigvecs_new)
        if np.allclose(eigvecs_new, eigvecs, rtol=tol):
            break
        eigvecs = eigvecs_new

    # compute the eigenvalues
    eigvals = np.diag(eigvecs.T @ A @ eigvecs)

    return eigvals, eigvecs


# In[60]:


def SVD(A):
    """
    Compute Singular Value Decomposition of matrix A using NumPy.

    Args:
        A: numpy.array, matrix to be decomposed

    Returns:
        U: numpy.array, matrix containing left singular vectors
        s: numpy.array, array containing singular values
        V_T: numpy.array, matrix containing right singular vectors (transposed)
    """
    # Compute the eigenvectors and eigenvalues of A*At or At*A, whichever is smaller
    if A.shape[0] < A.shape[1]:
        S = np.dot(A, A.T)
    else:
        S = np.dot(A.T, A)

    # eigvals, eigvecs = np.linalg.eig(S) #NOT ALLOWED
    eigvals, eigvecs = eig(S)

    # Sort the eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Compute the singular values and their reciprocals
    s = np.sqrt(eigvals)
    s_inv = np.zeros_like(A.T)
    np.fill_diagonal(s_inv, 1.0 / s)

    # Compute the left and right singular vectors
    U = np.dot(A, np.dot(eigvecs, s_inv))
    V_T = eigvecs.T

    if (len(s) != U.shape[0]):
        U = U[:, :len(s) - U.shape[0]]
    sigma = np.zeros([U.shape[1], V_T.shape[0]])

    for i in range(len(s)):
        sigma[i, i] = s[i]

    return U, s, sigma, V_T


def ReducedSVD(A, threshold=0, to_remove=0):
    U, s, sigma, V_trans = SVD(A)

    # While converting to python code we will convert into GUI asking-
    #       - Removal based on:-
    #       - 1. Hyper parameter
    #       - 2. Threshold

    # Removal based on hyper parameter
    if (to_remove < len(s) and to_remove > 0):
        s = s[:-to_remove]
        print(s)
        U = U[:, :-to_remove]
        V_trans = V_trans[:-to_remove, :]
        sigma = sigma[:-to_remove, :-to_remove]

    elif (to_remove < 0):
        print("The number of eigen values to be romved is Invalid!!")
        exit()

    # Removal based on threshold
    if (threshold < s[0] and threshold > 0):
        s = s[s >= threshold]
        print(s)
        U = U[:, :len(s)]
        V_trans = V_trans[:len(s), :]
        sigma = sigma[:len(s), :len(s)]

    elif (threshold < 0):
        print("Invalid threshold value!!")
        exit()

    return U, s, sigma, V_trans


# # Reduced SVD on Data
