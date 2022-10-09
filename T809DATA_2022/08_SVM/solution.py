# Author: Davide Parente
# Date: 9/10/2022
# Project: Support Vector Machines
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.
import sklearn.datasets

import tools
from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X, t = make_blobs(40,centers=2)
    clf = svm.SVC(C=1000,kernel='linear')
    clf.fit(X,t)
    plot_svm_margin(clf,X,t)
    plt.savefig("1_1_1.png")

def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''

    plt.subplot(1, num_plots, index)
    plot_svm_margin(svc,X,t)



def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    gamma=['scale',0.2,2]
    for i, g in enumerate(gamma):
        clf = svm.SVC(C=1000,kernel='rbf',gamma=g)
        clf.fit(X,t)
        #print("Numer of support vector for gamma",gamma[i],":",clf._n_support)
        #print(clf.get_params(deep=True))
        _subplot_svm_margin(clf,X,t,len(gamma),i+1)

    plt.savefig("1_1_3.png")


def _compare_C():
    plt.clf()
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    C_types = [1000,0.5,0.3,0.05,0.0001]
    for i, C in enumerate(C_types):
        clf = svm.SVC(C=C,kernel='linear')
        clf.fit(X,t)
        #print("Numer of support vector for C",C,":",clf._n_support)
        #print(clf.support_vectors_)
        _subplot_svm_margin(clf,X,t,len(C_types),i+1)
    plt.savefig("1_1_5.png")

def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train,t_train)
    y=svc.predict(X_test)
    r=[]
    r.append(accuracy_score(t_test,y))
    r.append(precision_score(t_test,y))
    r.append(recall_score(t_test,y))
    return r



def compare_train():
    category_kernel= ['linear','rbf','poly']
    res ={}
    (X_train, t_train), (X_test, t_test) = load_cancer()
    for x in category_kernel:
        svc = svm.SVC(C=100,kernel=x)
        res[x]=train_test_SVM(svc, X_train, t_train, X_test, t_test)
    return res

#Wine Dataset
def indep():
    wine= sklearn.datasets.load_wine()
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(
        wine.data, wine.target,
        test_size=0.3)
    category_kernel= ['linear','rbf','poly']
    res ={}
    for i,x in enumerate(category_kernel):
        plt.clf()
        r=[]
        svc = svm.SVC(C=100,kernel=x)
        svc.fit(X_train,t_train)
        y=svc.predict(X_test)
        r.append(accuracy_score(t_test,y))
        r.append(precision_score(t_test,y,average='macro'))
        r.append(recall_score(t_test,y,average='macro'))
        res[x]=r
    return res


def indep2():
    (X_train, t_train), (X_test, t_test)= tools.load_binary_iris()
    svc = svm.SVC(C=50,kernel='rbf')
    X_train = X_train[:,0:2]
    X_test= X_test[:,0:2]
    svc.fit(X_train,t_train)
    y=svc.predict(X_test)
    category_kernel= ['linear','rbf','poly']
    res ={}
    for i,x in enumerate(category_kernel):
        svc = svm.SVC(C=100,kernel=x)
        res[x]=train_test_SVM(svc, X_train, t_train, X_test, t_test)
        plot_svm_margin(svc,X_test,t_test)
        title= x+" kernel"
        plt.title(title)
        plt.show()
        plt.clf()
    return res


'''
_plot_linear_kernel()
_compare_gamma()
_compare_C()
(X_train, t_train), (X_test, t_test) = load_cancer()
svc = svm.SVC(C=1000)
print(train_test_SVM(svc, X_train, t_train, X_test, t_test))
x= compare_train()
for a in x:
    print(a,"kernel:")
    print("Accuracy score:", x[a][0])
    print("Precision score:", x[a][1])
    print("Recall score:", x[a][2])

print(indep())
print(indep2())
'''
