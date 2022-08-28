# Author: Davide Parente
# Date: 26/08/2022
# Project: 01_decision_trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    total = len(targets)
    class_probs = []
    for class_element in classes:
        counter = 0
        for target_element in targets:
            if class_element == target_element:
                counter += 1
        class_probs.append(counter / total)
    return class_probs


def split_data(
        features: np.ndarray,
        targets: np.ndarray,
        split_feature_index: int,
        theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:, split_feature_index] < theta, :]
    targets_1 = targets[features[:, split_feature_index] < theta]

    features_2 = features[features[:, split_feature_index] >= theta, :]
    targets_2 = targets[features[:, split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    C = 0
    for classes_element in classes:
        N_element = 0
        for target_element in targets:
            if target_element == classes_element:
                N_element += 1
        C += np.power((N_element / targets.shape[0]), 2)
    return 0.5* (1 - C)


def weighted_impurity(
        t1: np.ndarray,
        t2: np.ndarray,
        classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1,classes)
    g2 = gini_impurity(t2,classes)
    n = t1.shape[0] + t2.shape[0]
    return (t1.shape[0]*g1)/n + (t2.shape[0]*g2)/n



def total_gini_impurity(
        features: np.ndarray,
        targets: np.ndarray,
        classes: list,
        split_feature_index: int,
        theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (features_1, targets_1), (features_2, targets_2) = split_data(features,targets,split_feature_index,theta)
    if targets_1.shape[0] == 0 or targets_2.shape[0] == 0 :
        return 1
    return weighted_impurity(targets_1,targets_2,classes)


''' 

'''
def brute_best_split(
        features: np.ndarray,
        targets: np.ndarray,
        classes: list,
        num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None


    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min = float("inf")
        max = -float("inf")
        for j in range(features.shape[0]):
            if features[j][i]<min:
                min = features[j][i]
            if features[j][i]>max:
                max=features[j][i]
        thetas = np.linspace( min,max, num_tries+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            calculatedGiny = total_gini_impurity(features,targets,classes,i,theta)
            if calculatedGiny<best_gini :
                best_gini=calculatedGiny
                best_dim= i
                best_theta=theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            classes: list = [0, 1, 2],
            train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets), \
        (self.test_features, self.test_targets) = \
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        '''https://stackoverflow.com/questions/54884252/check-the-accuracy-of-decision-tree-classifier-with-python'''        #res_pred= self.tree.predict(self.test_features)
        score= self.tree.score(self.test_features, self.test_targets)
        #print(score)
        return score

    def plot(self):
        plot_tree(self.tree)
       # plt.show()
        plt.savefig('2_3_1.png')

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        x=[]
        y=[]
        plt.clf()
        for i in range(1,len(self.train_features)):
            self.tree.fit(self.train_features[:i], self.train_targets[:i])
            x.append(i)
            y.append(self.accuracy())
        plt.plot(x,y)
        plt.savefig('indep_1.png')

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        y_pred= self.guess()
        #print(y_pred)
        #print(self.test_targets)
        confusion_matrix= [[0,0,0],[0,0,0],[0,0,0]]
        for i in range(self.test_targets.shape[0]):
            confusion_matrix[self.test_targets[i]][y_pred[i]]+=1
        return confusion_matrix


'''
features, targets, classes = load_iris()

print("----------------- TEST 1.1 -----------------")
print(prior([0, 0, 1], [0, 1]))
print(prior([0, 2, 3, 3], [0, 1, 2, 3]),"\n")


print("----------------- TEST 1.2 -----------------")
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
print(len(f_1), "-  90, ", len(f_1)==90 )
print(len(f_2), "-  60, ", len(f_2)==60,"\n")

print("----------------- TEST 1.3 -----------------")
print(gini_impurity(t_1, classes), "-  0.2517283950617284,  ", gini_impurity(t_1, classes)==0.2517283950617284)
print(gini_impurity(t_2, classes), "-  0.1497222222222222,  ", gini_impurity(t_2, classes)==0.1497222222222222,"\n")


print("----------------- TEST 1.4 -----------------")
print( weighted_impurity(t_1, t_2, classes),"-   0.2109259259259259,  ", weighted_impurity(t_1, t_2, classes)== 0.2109259259259259, "\n")

print("----------------- TEST 1.5 -----------------")
print(total_gini_impurity(features, targets, classes, 2, 4.65),"-   0.2109259259259259,  ", weighted_impurity(t_1, t_2, classes)== 0.2109259259259259,"\n")

print("----------------- TEST 1.6 -----------------")
res=brute_best_split(features,targets, classes,30)
print(res,"-  (0.16666666666666666, 2, 1.9516129032258065)" , res==np.asarray([(0.16666666666666666, 2, 1.9516129032258065)]),"\n")


p= IrisTreeTrainer(features,targets)
print("Training the model")
p.train()

print("----------------- TEST 2.2 -----------------")
s=p.accuracy()
print(s)

print("----------------- TEST 2.3 -----------------")
p.plot()
print("image -> 2_3_1.png")

print("----------------- TEST 2.4 -----------------")
s= p.guess()
print(s)
print("----------------- TEST 2.5 -----------------")
print(p.confusion_matrix())

print("------------- INDEPENDENT PART -------------")
dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
dt.plot_progress()
print("image -> indep_1.png")
'''
