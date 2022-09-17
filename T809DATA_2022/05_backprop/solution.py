from typing import Union
import numpy as np

from tools import load_iris, split_train_test


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
    aOne= []
    z1= []
    for i in range(M):
        aOne.append(sum(z0*W1[:,i]))
        z1.append(sigmoid(aOne[i]))
    z1.insert(0,1)
    aTwo=[]
    y=[]
    for i in range(K):
        aTwo.append(sum(z1*W2[:,i]))
        y.append(sigmoid(aTwo[i]))
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

    #Play with index
    W=W2.T
    for j in range(len(aOne)):
        value=0
        for i in range(K):
            value+=W[i,j+1]*delta_k[i]
        delta_y.append(d_sigmoid(aOne[j])*value)
    '''
    print("Ci" ,delta_y)

    print(len(delta_y))
    print("a1  ", len(aOne), "  a2 ",len(aTwo))
    print("W1 ",W1.shape[0]," ",W1.shape[1])

    print("W2 ",W2.shape[0]," ",W2.shape[1])
    print("z0 ",len(z0), "  z1 ",len(z1))
    '''
    for i in range(dE1.shape[0]):
        for j in range(dE1.shape[1]):
            dE1[i][j] = delta_y[j]*z0[i]
    #print(np.asarray(dE1))

    for i in range(dE2.shape[0]):
        for j in range(dE2.shape[1]):
            dE2[i][j] = delta_k[j]*z1[i]
    #print(np.asarray(dE2))
    return y_k,dE1,dE2
    #print(result)
    #print(y_k)


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
        sum=0
        errors=0
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
            for a in range(len(target_y)):
                sum+=target_y[a]*np.log(y[a])+(1-target_y[a])*np.log(1-y[a])
        E_total.append(-sum)
        misclassification_rate.append(errors/X_train.shape[0])
        W1 = W1- eta * dE1_total/ X_train.shape[0]
        W2 = W2- eta * dE2_total/ X_train.shape[0]

    print(E_total)
    print(misclassification_rate)
    print(results)
    print(t_train)
    return [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), E_total]
    #print(W1)
    #print(W2)



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
    ...

print("----TEST 1.1-----")
print(sigmoid(0.5) == 0.6224593312018546)
print(d_sigmoid(0.2) == 0.24751657271185995)


print("----TEST 1.2 -----")
a =perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))
print(a[0]==1.0799999999999998 , " ", a[1]==0.7464939833376621)
b= perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))
print(0.18000000000000005 == b[0], " ", b[1] == 0.5448788923735801)


print("----TEST1.3 ------")


features, targets, classes = load_iris()
for e in features:
    if(e[0]== 6.3 and e[1] == 2.5 and e[2]==4.9 and e[3]==1.5):
        x=e
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
#x = train_features[0, :]
np.random.seed(1234)
K=3
M=10
D=4
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
y, z0, z1, a1, a2 =ffnn(x, M, K, W1, W2)
print("y=",y)
print("z0=",z0)
print("z1=",z1)
print("a0=",a1)
print("a1=",a2)




print("----TEST 1.4----")
#Remember : Non conosci la classe siccome hai saccheggiato il valore ciclatelo (target y)
np.random.seed(42)

K = 3  # number of classes
M = 6
D = train_features.shape[1]
target_y = np.zeros(K)
target_y[targets[0]] = 1.0
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

#y, dE1, dE2 =
backprop(x, target_y, M, K, W1, W2)

print(W1)
print(W2)


print("----- TEST 1.4 -----")

np.random.seed(1234)
(train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
K = 3  # number of classes
M = 6
D = train_features.shape[1]
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses =train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)


