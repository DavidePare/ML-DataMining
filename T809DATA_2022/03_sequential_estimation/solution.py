# Author: Davide Parente
# Date: 9/9/2022
# Project: Sequential Estimation
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



def _plot_sequence_estimate():
    #data= gen_data(300,3,np.array([0,0,0]),1)
    data = gen_data(500,3, np.array([0, 1, -1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.savefig("1_5_1.png")


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
    plt.savefig("1_6_1.png")


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    value=end_mean-start_mean
    numberOfElements=20
    #Generation of the first element
    data = gen_data(numberOfElements,k,start_mean,var)
    #Calculate how many times the gen_data must be done
    rangevalue= n/numberOfElements
    for i in range(1,int(rangevalue)+1):
        data=np.append(data,gen_data(numberOfElements,k,
        start_mean+(value*(i/(n/numberOfElements))),var),axis=0)
    return data



def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    data= gen_changing_data(500,3,np.array([0,1,-1]),np.array([1,-1,0]),1)
    y=np.mean(data)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
    elements= []
    for e in estimates:
        elements.append(_square_error(y,np.mean(e)))
    plt.plot(elements)
    plt.savefig("indep_1.png")
    plt.clf()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.savefig("indep_2.png")






