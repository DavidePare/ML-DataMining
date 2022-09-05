# Author: 
# Date:
# Project: 
# Acknowledgements: 
#
import tools
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    np.random.seed(1234)
    standard_deviation=np.power(var,2)
    identity_matrix= np.identity(k)
    return np.random.multivariate_normal(mean,identity_matrix*standard_deviation,n)



def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return (mu+((1/n)*(x-mu)))
    #return (n*mu+x)/(n+1)



def _plot_sequence_estimate():
    data = gen_data(3000,3, np.array([0, 1, -1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.power(y-y_hat,2)


def _plot_mean_square_error():
    data = gen_data(100,3, np.array([0, 1, -1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    y=np.mean(data)
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i,:], i+1))
    elements= []
    for e in estimates:
        elements.append(_square_error(y,np.mean(e)))
    plt.plot(elements)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    ...


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...


print("-------TEST 1.1---------")
x=gen_data(2, 3, np.array([0, 1, -1]), 1.3)
print(x)
print("Correct Result \n [[ 0.61286571, -0.5482684 ,  0.86251906],\n [-0.40644746,  0.06323465,  0.15331182]]")
#m=np.asmatrix([[ 0.61286571, -0.5482684 ,  0.86251906],[-0.40644746,  0.06323465,  0.15331182]])
#print(m==np.asmatrix(x))

print("-------TEST 1.1---------")
x=gen_data(5, 1, np.array([0.5]), 0.5)
print(x)
print("Correct Result \n [[ 0.73571758],\n[-0.09548785],\n[ 1.21635348],\n[ 0.34367405],\n[ 0.13970563]]")
#m=np.asmatrix([[ 0.73571758],[-0.09548785],[ 1.21635348],[ 0.34367405],[ 0.13970563]])
#print(np.asmatrix(x)==m)

print("-------TEST1.2-------")
x=gen_data(300,3, np.array([0, 1, -1]),np.sqrt(3))
tools.scatter_3d_data(x)
tools.bar_per_axis(x)


print("-------TEST1.4------")
mean = np.mean(x, 0)
new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
print(x)
print(update_sequence_mean(mean, new_x, x.shape[0]))

_plot_sequence_estimate()

print("-------TEST1.5------")
_plot_mean_square_error()
