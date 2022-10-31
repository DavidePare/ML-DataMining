# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    a=np.zeros(X.shape)
    sigma=np.std(X)
    mean= np.mean(X)
    for i in range(X.shape[0]):
        a[i]=(X[i]-mean)/sigma
    return a

def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    stand=standardize(X)
    x = np.zeros(X.shape[0])
    y = np.zeros(X.shape[0])
    for s,a in enumerate(stand):
        x[s]= a[i]
        y[s]= a[j]
    plt.scatter(x,y)


def _scatter_cancer():
    X, y = load_cancer()
    ...


def _plot_pca_components():
    ...
    X, y = load_cancer()
    for i in range(...):
        plt.subplot(5, 6, ...)
        ...
    plt.show()


def _plot_eigen_values():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    ...
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))

X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
scatter_standardized_dims(X, 0, 2)
plt.show()
