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
    for k in range(R.shape[1]):
        n_R=0
        for n in range(R.shape[0]):
            u[k]+=R[n][k]*X[n]
            n_R+=R[n][k]
        u[k]=u[k]/n_R
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

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]
    Js=[]
    # Then we have to "de-standardize" the prototypes
    for x in range(num_its):
        dist=distance_matrix(X_standard,Mu)
        R=determine_r(dist)
        Mu=update_Mu(Mu,X_standard,R)
        Js.append(determine_j(R,dist))


    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu,R,Js


def _plot_j():
    X, y, c = load_iris()
    Mu,R,Js=k_means(X, 4, 10)
    plt.plot(Js)
    plt.savefig("1_6_1.png")


def _plot_multi_j():
    X, y, c = load_iris()
    k= [2,3,5,10]
    figure, axis = plt.subplots(2, 2)
    for i,element in enumerate(k):
        Mu,R,Js=k_means(X, element, 10)
        axis[int(i/2), i%2].plot(Js)
        axis[int(i/2), i%2].set_title("k="+str(element))
    figure.tight_layout()
    plt.savefig("1_7_1.png")


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
    Mu,R,Js= k_means(X,len(classes),num_its)
    bestCluster = np.zeros((len(classes),len(classes)))

    for i in range(t.shape[0]):
        bestCluster[np.argmax(R[:][i])][t[i]]+=1
    ClusterFinal=np.zeros(len(classes))
    for i in range(len(classes)):
        ClusterFinal[i]=np.argmax(bestCluster[i])
    prediction= np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        prediction[i]=ClusterFinal[np.argmax(R[i])]
    return prediction

def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    prediction=k_means_predict(X, y, c, 5)
    acc=0
    for i in range(prediction.shape[0]):
        if(y[i]==prediction[i]):
            acc+=1
    confusion_matrix = np.zeros((len(c),len(c)))
    for i in range(len(y)):
        confusion_matrix[y[i]][int(prediction[i])]+=1
    print("Accuracy: "+str(acc/len(y)))
    print("Confusion Matrix:")
    print(confusion_matrix)

def _my_kmeans_on_image():
    num_cluster= [2,5,10,20]
    for x in num_cluster:
        plot_image_clusters(x)




def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    plt.clf()
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters).fit(image)

    plt.subplot(1,2,1)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(1,2,2)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.savefig("2_1_"+str(n_clusters)+".png")


def _gmm_info():
    X, y, c = load_iris()
    g = GaussianMixture(n_components=3).fit(X,y)
    return g.means_, g.covariances_, g.weights_


def _plot_gmm():
    X, y, c = load_iris()
    g = GaussianMixture(n_components=3).fit(X,y)
    predictions = g.predict(X)
    plot_gmm_results(X,predictions,g.means_,g.covariances_)

def indep1(
    X: np.ndarray,
    t: np.ndarray,
    k: int, #Number of clusters
    classes: list,
    num_its: int
    ) -> np.ndarray:
    Mu,R,Js= k_means(X,k,num_its)
    bestCluster = np.zeros((k,len(classes)))
    colors=['blue','orange','green','olive','purple','brown','pink','yellow','gray','cyan']
    for i in range(t.shape[0]):
        bestCluster[np.argmax(R[:][i])][t[i]]+=1
    ClusterFinal=np.zeros(k)
    for i in range(k):
        ClusterFinal[i]=np.argmax(bestCluster[i])
    prediction= np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        x, y = X[i,0],X[i,2]
        prediction[i]=ClusterFinal[np.argmax(R[i])]
        if(prediction[i]==t[i]):
            plt.scatter(x, y,c=colors[np.argmax(R[i])],linewidths=2)
        else:
            plt.scatter(x, y,c=colors[np.argmax(R[i])],edgecolors='red',linewidths=2)
    print("Accuracy:",accuracy_score(t,prediction))
    print("Confusion matrix:")
    print(confusion_matrix(t,prediction))
    plt.savefig("indep.png")
    return prediction

def indep2(
    X: np.ndarray,
    t: np.ndarray,
    k: int, #Number of clusters
    classes: list,
    num_its: int
    ) -> np.ndarray:
    Mu,R,Js= k_means(X,k,num_its)
    bestCluster = np.zeros((k,len(classes)))
    for i in range(t.shape[0]):
        bestCluster[np.argmax(R[:][i])][t[i]]+=1
    ClusterFinal=np.zeros(k)
    for i in range(k):
        ClusterFinal[i]=np.argmax(bestCluster[i])
    prediction= np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        prediction[i]=ClusterFinal[np.argmax(R[i])]
    print("Accuracy:",accuracy_score(t,prediction))
    print("Confusion matrix:")
    print(confusion_matrix(t,prediction))
    return prediction

def plot_image_clusters_walterWhite(
        path: str = './images/Walter_White_S5B.png'
    ):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    plt.clf()
    print(path)
    image, (w, h) = image_to_numpy(path)
    n_clusters=[2,5,10,20,30]
    plt.subplot(1,6,1)
    plt.imshow(image.reshape(w, h, 3))
    for i,x in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=x).fit(image)
        plt.subplot(1,6,i+2)
        # uncomment the following line to run
        plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")

    plt.title("Walter White")
    plt.savefig("indep2.png")

#X, y, c = load_iris()

#indep2(X,y,50,c,10)
#indep2(X,y,80,c,10)


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
update_Mu(Mu, X, R)

print("--- Test 1.5 ---")
X, y, c = load_iris()
#print(k_means(X, 4, 10))


print("--- Test 1.6 ---")
_plot_j()

print("--- Test 1.7 ---")
_plot_multi_j()

print("--- Test 1.9 ---")
X, y, c = load_iris()
k_means_predict(X, y, c, 5)

print("--- Test 1.10 ---")
_iris_kmeans_accuracy()

print("--- Test 2.1 ---")
#_my_kmeans_on_image()
print("--- Test 3.1")
print(_gmm_info())


print("--- Test 3.2 ---")
_plot_gmm()

_iris_kmeans_accuracy()

_my_kmeans_on_image()


#Funziona
print("--- ExtraTest ---")
X, y, c = load_iris()
np.random.seed(42)
Mu, R, Js = k_means(X, 4, 10)
print(Js)
