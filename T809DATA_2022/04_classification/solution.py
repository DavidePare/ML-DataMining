# Author: 
# Date:
# Project: 
# Acknowledgements: 
#
import help
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    count=0
    value=[0,0,0,0]
    for i,feature in enumerate(features):
        if targets[i] == selected_class:
            value+=feature
            count+=1
    return value/count

def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    value=[]
    for i,feature in enumerate(features):
        if targets[i] == selected_class:
            value.append(feature)
    return np.cov(value,rowvar=False)

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature[:])
    #print(multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature[1]))
    #print(multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature[5:5]))
    #print(multivariate_normal(mean=class_mean, cov=class_covar).pdf([100, 100]))
    #help.pdf()

def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''

    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([likelihood_of_class(test_features[i],means[0],covs[0]),likelihood_of_class(test_features[i],means[1],covs[1]),likelihood_of_class(test_features[i],means[2],covs[2])])

    #print(len(classes)*test_features.shape[0])
    #print(len(likelihoods))
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.asarray([np.argmax(element) for element in likelihoods])

def mean_training(
    targets: np.ndarray
) -> np.ndarray:
    m=[0,0,0]
    for x in targets:
        m[x]+=1
    for i in range(0,len(m)):
        m[i]= m[i]/targets.shape[0]
    return m

def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    m = mean_training(train_targets)
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    aposteriori = []
    print(m)
    print("Mmmm  ",means)
    print(type(means))
    for i in range(test_features.shape[0]):
        aposteriori.append([m[0]*likelihood_of_class(test_features[i],means[0],covs[0]),
                            m[1]*likelihood_of_class(test_features[i],means[1],covs[1]),
                            m[2]*likelihood_of_class(test_features[i],means[2],covs[2])])
    #print(len(classes)*test_features.shape[0])


    return np.array(aposteriori)

def accurancy(
    predict_result: np.ndarray,
    targets: np.ndarray
):

    occurences=0
    for i in range(0,predict_result.shape[0]):
        if(predict_result[i]==targets[i]):
            occurences+=1
    print(predict_result.shape[0])
    print(occurences)
    return occurences/ predict_result.shape[0]

def confusion_matrix_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    classes: list
):
    likelihood=maximum_likelihood(train_features, train_targets, test_features, classes)
    y_pred=predict(likelihood)
    confusion_matrix= [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(test_targets.shape[0]):
        confusion_matrix[test_targets[i]][y_pred[i]]+=1
    return confusion_matrix

def confusion_matrix_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    classes: list
):
    aposteriori=maximum_aposteriori(train_features, train_targets, test_features, classes)
    y_pred=predict(aposteriori)
    confusion_matrix= [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(test_targets.shape[0]):
        confusion_matrix[test_targets[i]][y_pred[i]]+=1
    return confusion_matrix

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)
print("-----TEST1.1-----")
print(mean_of_class(train_features, train_targets, 0))
print()
'''
print("-----TEST1.2-----")
print(covar_of_class(train_features, train_targets, 0))
print()

print("-----TEST1.3-----")
class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
print(likelihood_of_class(test_features[0, :], class_mean, class_cov))


print("-----TEST1.4-----")
print(maximum_likelihood(train_features, train_targets, test_features, classes))
'''




print("-----TEST1.5-----")
likelihood=maximum_likelihood(train_features, train_targets, test_features, classes)

print(predict(likelihood))
print(test_targets[:])

print("-----TEST2.1-----")
aposteriori=maximum_aposteriori(train_features, train_targets, test_features, classes)
print(predict(aposteriori))


f1,t1=features[:25],targets[:25]
(train_features, train_targets), (a, b)\
    = split_train_test(features, targets, train_ratio=0.8)

#f1,t1=features[50:75],targets[50:75]
#(train_features, train_targets), (a, b)\
#    = split_train_test(features, targets, train_ratio=0.8)
f1,t1=features[70:95],targets[70:95]
for i in range(0,50):
    #= np.array(train_features,f1)
    train_features= np.concatenate((train_features,f1),axis=0)
    train_targets=np.append(train_targets,t1)
#f1,t1=features[120:140],targets[120:140]
for i in range(0,50):
    #= np.array(train_features,f1)
    train_features= np.concatenate((train_features,f1),axis=0)
    train_targets=np.append(train_targets,t1)
'''
f1,t1=features[100:145],targets[50:145]
for i in range(0,1500):
    #= np.array(train_features,f1)
    train_features= np.concatenate((train_features,f1),axis=0)
    train_targets=np.append(train_targets,t1)

print(train_features[:250])
'''
print(train_targets.shape[0])
print(train_features.shape[0])

(train_features, train_targets),(test_features, test_targets)\
    = split_train_test(train_features, train_targets, train_ratio=0.8)


print(train_targets.shape[0])
print(train_features.shape[0])

print(train_targets[:500])
likelihood=maximum_likelihood(train_features, train_targets, test_features, classes)
#print("-----TEST2.1-----")
aposteriori=maximum_aposteriori(train_features, train_targets, test_features, classes)
#print(predict(aposteriori))
#print(test_targets[:])
#print("-----TEST2.2.1-----")
print(accurancy(predict(likelihood),test_targets))
print(accurancy(predict(aposteriori),test_targets))


'''
for i in range(0,7):
    likelihood=maximum_likelihood(train_features, train_targets, test_features, classes)
    #print("-----TEST2.1-----")
    aposteriori=maximum_aposteriori(train_features, train_targets, test_features, classes)
    #print(predict(aposteriori))
    #print(test_targets[:])
    #print("-----TEST2.2.1-----")
    print(accurancy(predict(likelihood),test_targets))
    print(accurancy(predict(aposteriori),test_targets))

'''
'''

print("-----TEST2.2.1-----")
print(accurancy(predict(likelihood),test_targets))
print(accurancy(predict(aposteriori),test_targets))
print("-----TEST2.2.2-----")

print(confusion_matrix_likelihood(train_features,train_targets,test_features,test_targets,classes))
print(confusion_matrix_aposteriori(train_features,train_targets,test_features,test_targets,classes))
'''
