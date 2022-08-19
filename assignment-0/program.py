import numpy as np
import matplotlib.pyplot as plt

def normal_prob(x,sigma,mu):
    return (1/np.sqrt(2*np.pi*np.power(sigma,2)))*np.exp(-np.power((x-mu),2)/(2*np.power(sigma,2)))



def plot_normal(sigma, mu, x_start, x_end):
    return None
    #np.linspace(x_start,x_end)


def normal_mixture(x, sigmas, mus, weights):
    result=0
    for i in range(len(sigmas)):
        result= result+ ( weights[i]/np.sqrt(2*np.pi*np.power(sigmas[i],2))*np.exp(-np.power((x-mus[i]),2)/(2*np.power(sigmas[i],2))))
    return result
       



def sample_gaussian_mixture(sigmas, mus, weights, n_samples):
    return None


