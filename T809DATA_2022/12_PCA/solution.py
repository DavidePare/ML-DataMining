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

def notstand(
    X: np.ndarray,
    i: int,
    j: int,
):
    x = np.zeros(X.shape[0])
    y = np.zeros(X.shape[0])
    for s,a in enumerate(X):
        x[s]= a[i]
        y[s]= a[j]
    plt.scatter(x,y)

def _scatter_cancer():
    plt.clf()
    plt.figure(figsize=(12,10))
    X, y = load_cancer()
    print(X.shape[1])
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i+1)
        notstand(X,0,i)

    plt.savefig("1_3_1.png")

def _plot_pca_components():

    plt.clf()
    pca= PCA()
    X, y = load_cancer()
    X_std=standardize(X)
    plt.figure(figsize=(20,18))
    pca.fit_transform(X_std)
    for i in range(pca.components_.shape[1]):
        plt.subplot(5, 6, i+1)
        plt.plot(pca.components_[i])
        plt.title("PCA "+str(i+1))
    plt.savefig("2_1_1.png")


def _plot_eigen_values():

    plt.clf()
    pca= PCA()
    X, y = load_cancer()
    X_std=standardize(X)
    pca.fit_transform(X_std)
    print(pca.explained_variance_)
    plt.plot(pca.explained_variance_)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig("3_1_1.png")


def _plot_log_eigen_values():
    plt.clf()
    pca= PCA()
    X, y = load_cancer()
    X_std=standardize(X)
    pca.fit_transform(standardize(X_std))
    plt.plot(np.log10(pca.explained_variance_))

    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.savefig("3_1_2.png")


def _plot_cum_variance():
    plt.clf()
    pca= PCA()
    X, y = load_cancer()
    X_std=standardize(X)
    pca.fit_transform(standardize(X_std))
    plt.plot(np.cumsum(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.savefig("3_1_3.png")

'''
print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))

X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
_scatter_cancer()
_plot_pca_components()

_plot_eigen_values()
_plot_log_eigen_values()
_plot_cum_variance()
'''
