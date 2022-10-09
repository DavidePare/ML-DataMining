# Author: Davide Parente
# Date: 09/10/2022
# Project: Linear Models for Regression
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt
import scipy

from tools import load_regression_iris
from scipy.stats import multivariate_normal
import sklearn.datasets as datasets



def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''

    res= np.zeros([features.shape[0],mu.shape[0]])
    for i in range(features.shape[0]):
        for j in range(mu.shape[0]):
            res[i][j]=multivariate_normal.pdf(features[i,:],mu[j,:],sigma)


    return res



def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    plt.plot(fi[:,0])
    plt.savefig("plot_1_2_OneDimension")
    plt.clf()
    plt.plot(fi)
    plt.savefig("plot_1_2")


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''

    X=np.matmul(np.linalg.inv(np.matmul(fi.T,fi)+(lamda*np.identity(fi.shape[1]))),np.matmul(fi.T,targets))
    return X

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    ...
    A=mvn_basis(features,mu,sigma)
    value = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        value[i]+=np.matmul(w.T,A[i,:])

    return value

def pred(targets,predict):
    errors= []
    plt.clf()
    for i in range(len(targets)):
        errors.append(np.power(targets[i]-predict[i],2))
    plt.plot(errors)
    plt.savefig("squareerror.png")
    plt.clf()

def acc(targets,predict):
    t=0
    for i in range(len(predict)):
        if(targets[i]==predict[i]):
            t+=1
    return t/len(predict)


'''
print("---TEST 1.1---")
X, t = load_regression_iris()
N, D = X.shape
M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)
print(fi)
lamda = 0.001
wml = max_likelihood_linreg(fi, t, lamda)
print(wml)
prediction = linear_model(X, mu, sigma, wml)

_plot_mvn()

pred(t,prediction)
'''
X, t = load_regression_iris()
N, D = X.shape
M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)
lamda = 0.001
wml = max_likelihood_linreg(fi, t, lamda)
prediction = linear_model(X, mu, sigma, wml)
print("W",prediction)
#Diabets Dataset
def indep1():
    diab= datasets.load_diabetes()
    X, t = diab.data, diab.target
    N, D = X.shape
    M, sigma = 100, 1
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    lamda = 0.1
    wml = max_likelihood_linreg(fi, t, lamda)
    prediction = linear_model(X, mu, sigma, wml)
    pred(t,prediction)

#blob make
def indep2():
    X, t = datasets.make_blobs(40,centers=2)
    N, D = X.shape
    M, sigma = 10, 2
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    prediction = linear_model(X, mu, sigma, wml)
    pred(t,prediction)

_plot_mvn()
