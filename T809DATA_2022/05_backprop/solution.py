# Author: Davide Parente
# Date: 25/09/2022
# Project: 05 Neural Network
# Acknowledgements:
#
from typing import Union
import numpy as np

from tools import load_iris, split_train_test
import matplotlib.pyplot as plt


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if(x<-100):
        return 0
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return np.exp(-x)/np.power(1+np.exp(-x),2)


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    sum=0
    for i in range(len(x)):
        sum+= x[i]*w[i]
    return sum, sigmoid(sum)



def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0=np.insert(x,0,1)
    aOne= np.zeros(M)
    z1=np.zeros(M)
    for i in range(M):
        aOne[i],z1[i]=perceptron(z0,W1[:,i])
    z1=np.insert(z1,0,1)
    aTwo=np.zeros(K)
    y=np.zeros(K)
    for i in range(K):
        aTwo[i],y[i]=perceptron(z1,W2[:,i])
    return y,z0,z1,aOne,aTwo

def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''

    y_k,z0,z1,aOne,aTwo=ffnn(x,M,K,W1,W2)
    dE1= np.zeros((W1.shape[0],W1.shape[1]))
    dE2= np.zeros((W2.shape[0],W2.shape[1]))
    delta_k = y_k -target_y
    delta_y=[]

    W=W2.T
    for j in range(len(aOne)):
        value=0
        for i in range(K):
            value+=W[i,j+1]*delta_k[i]
        delta_y.append(d_sigmoid(aOne[j])*value)

    for i in range(dE1.shape[0]):
        for j in range(dE1.shape[1]):
            dE1[i][j] = delta_y[j]*z0[i]

    for i in range(dE2.shape[0]):
        for j in range(dE2.shape[1]):
            dE2[i][j] = delta_k[j]*z1[i]
    return y_k,dE1,dE2



def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    E_total=[]
    misclassification_rate= []
    results = []
    for j in range(iterations):
        sum=np.zeros(X_train.shape[0])
        errors=0
        aii=0
        dE1_total= np.zeros((W1.shape[0],W1.shape[1]))
        dE2_total= np.zeros((W2.shape[0],W2.shape[1]))
        for i,e in enumerate(X_train):
            target_y = np.zeros(K)
            target_y[t_train[i]] = 1.0
            y,dE1,dE2=backprop(e,target_y,M,K,W1,W2)
            position=np.argmax(y)
            dE1_total+=dE1
            dE2_total+=dE2
            if(j==iterations-1):
                results.append(position)
            if(target_y[position]!= 1):
                errors+=1
            sum[i]=np.sum(target_y*np.log(y)+(1-target_y)*np.log(1-y))
        E_total.append(-np.mean(sum))
        misclassification_rate.append(errors/X_train.shape[0])
        W1 = W1- eta * dE1_total/ X_train.shape[0]
        W2 = W2- eta * dE2_total/ X_train.shape[0]

    return W1,W2,np.asarray(E_total),misclassification_rate,results



def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    results=[]
    for i in range(X.shape[0]):
        y,z,z1,a1,a2=ffnn(X[i],M,K,W1,W2)
        results.append(np.argmax(y))
    return np.asarray(results)

def tester(features,targets):
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    #np.random.seed(1234)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses =train_nn(train_features, train_targets, M, K, W1, W2, 500, 0.1)
    accuracy_ofTrainer= accurancy(np.asarray(last_guesses),train_targets)
    y_predicted=test_nn(test_features,M,K,W1tr,W2tr)
    accuracy_ofTester= accurancy(np.asarray(y_predicted),test_targets)
    print("Accuracy of trainer:",accuracy_ofTrainer)
    print("Accuracy of tester:",accuracy_ofTester)
    matrixOne=confusion_matrix(np.asarray(last_guesses),train_targets)
    matrixTwo=confusion_matrix(np.asarray(y_predicted),test_targets)
    print("Confusion matrix of the trainer:")
    print(matrixOne)
    print("Confusion matrix of the tester:")
    print(matrixTwo)
    plot_ETotal(Etotal)
    plot_misclassification_rate(misclassification_rate)

def independet_part_changing_training_number(features,targets):
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    accuracy_ofTrainer = []
    x_plot= []
    accuracy_ofTester = []

    # Configuring the learning rate

    for i in range(1,50):
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses =train_nn(train_features, train_targets, M, K, W1, W2, 500, 0.005*i)
        accuracy_ofTrainer.append(accurancy(np.asarray(last_guesses),train_targets))
        y_predicted=test_nn(test_features,M,K,W1tr,W2tr)
        accuracy_ofTester.append(accurancy(np.asarray(y_predicted),test_targets))
        if(i%2==0):
            print("Case analyzed=",i)
            x_plot.append(0.005*i)

    x = np.linspace(x_plot[0], x_plot[len(x_plot)-1], 49)


    plt.xlabel('LearningRate')
    plt.ylabel('TesterAccuracy')
    plt.plot(x,accuracy_ofTester)
    plt.title("Analysis of changing learning rate")

    plt.savefig("indep_learningrate.png",dpi=300)

    plt.clf()

    M=1
    x_plot= []
    accuracy_ofTester= []
    for i in range(0,24):
        M+=1
        W1 = 2 * np.random.rand(D + 1, M) - 1
        W2 = 2 * np.random.rand(M + 1, K) - 1
        x_plot.append(M)
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses =train_nn(train_features, train_targets, M, K, W1, W2, 100, 0.2)
        accuracy_ofTrainer.append(accurancy(np.asarray(last_guesses),train_targets))
        y_predicted=test_nn(test_features,M,K,W1tr,W2tr)
        accuracy_ofTester.append(accurancy(np.asarray(y_predicted),test_targets))
        if(i%10==0):
            print("Case analyzed=",i)

    plt.xticks(range(len(x_plot)), x_plot)
    plt.plot(accuracy_ofTester)
    plt.title("Analysis of changing size of the hidden layer")
    plt.xlabel("Number of neurons")
    plt.ylabel("Tester Accuracy")
    plt.savefig("indep_numberNeurons.png")


def accurancy(
    y_calucated: np.ndarray,
    y_test: np.ndarray
) -> float:
    correctness=0
    for i in range(y_calucated.shape[0]):
        if(y_calucated[i]== y_test[i]):
            correctness+=1
    return correctness/y_test.shape[0]

def confusion_matrix(
    y_pred: np.ndarray,
    target: np.ndarray
)  -> np.matrix:
    confusion_matrix= [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(target.shape[0]):
        confusion_matrix[target[i]][y_pred[i]]+=1
    return np.asmatrix(confusion_matrix)


def plot_ETotal(
    E_total : np.ndarray
):
    plt.plot(E_total)
    plt.savefig("eTotal.png")
    plt.clf()

def plot_misclassification_rate(
    misclassification_rate : np.ndarray

):
    plt.plot(misclassification_rate)
    plt.savefig("misclassification.png")
    plt.clf()


