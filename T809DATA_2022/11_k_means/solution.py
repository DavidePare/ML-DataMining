# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum = 0
    for i in range(x.shape[0]):
        sum += np.power(x[i] - y[i], 2)
    return np.sqrt(sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    distances = np.zeros((X.shape[0],Mu.shape[0]))
    for i,a in enumerate(X):
        distances[i]=euclidian_distances(a,Mu)
    return distances


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    r= np.zeros(dist.shape)
    for i,x in enumerate(dist):
        r[i][np.argmin(x)]=1
    return r

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function

    sum=0
    print(R.shape)
    u= np.zeros(R.shape[1])
    print(dist)
    for i in range(R.shape[1]):
        nCluster=np.sum(R[:,i])
        x=(R[:][i]*dist[i])
        print(x)
        u[i]= np.sum(x)/nCluster
    print("U",u)
    sum=0
    for n in range(R.shape[0]):
        for k in range(R.shape[1]):
            sum+=R[n][k]*np.power((np.abs(dist[n][k]-u[k])),2)*(1/np.sum(R[:,k]))
    return sum
    '''
    sum=0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if(R[i][j]==1):
                sum+=dist[i][j]
    return sum/R.shape[0]

def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    u= np.zeros(Mu.shape)
    '''
    for i in range(R.shape[1]):
        nCluster=np.sum(R[:,i])
        x=(R[:][i]*X[:][i])
        print(x)
        u[i]= np.sum(x)/nCluster
    print("U",u)
    '''
    for n in range(R.shape[0]):
        for k in range(R.shape[1]):
            u[:][k]+=R[n][k]*np.power((np.abs(X[n])),2)*(1/np.sum(R[:,k]))
            #if I put mu_k non riporta

    return u


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.
    #print(X_standard.shape)
    #R=determine_r(X_standard)
    #KMeans.fit()

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]
    dist=distance_matrix(X_standard,Mu)
    R=determine_r(dist)
    print(determine_j(R,dist))
    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean
        for x in range(num_its):
            dist=distance_matrix(X_standard,Mu)
        update_Mu(Mu,X,R)
        print(determine_j(R,dist))

    ...


def _plot_j():
    ...


def _plot_multi_j():
    ...


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    ...


def _iris_kmeans_accuracy():
    ...


def _my_kmeans_on_image():
    ...


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    ...
    plt.subplot('121')
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot('122')
    # uncomment the following line to run
    # plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    ...


def _plot_gmm():
    ...


print("---TEST 1.1---")
a = np.array([
    [1, 0, 0],
    [4, 4, 4],
    [2, 2, 2]])
b = np.array([
    [0, 0, 0],
    [4, 4, 4]])
print(distance_matrix(a, b))

print("---Test 1.2---")
dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
print(determine_r(dist))


print("--- TEST 1.3 ---")
dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
R = determine_r(dist)
print(determine_j(R, dist))
'''
print("--- TEST 1.4 ---")
X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])
Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])
R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])
print(update_Mu(Mu, X, R))

print("--- Test 1.5 ---")
X, y, c = load_iris()
k_means(X, 4, 10)
'''
