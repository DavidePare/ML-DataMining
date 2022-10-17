import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)

from collections import OrderedDict


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train,self.t_train)

    def prediction(self) -> np.ndarray:
        return self.classifier.predict(self.X_test)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        y_pred= self.prediction()
        return confusion_matrix(self.t_test,y_pred)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        y_pred= self.prediction()
        return accuracy_score(self.t_test,y_pred)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        y_pred= self.prediction()
        return precision_score(self.t_test,y_pred)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        y_pred= self.prediction()
        return recall_score(self.t_test,y_pred)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        #cross_val_score()
        #TODO
        return cross_val_score(self.classifier,self.X_test,self.t_test)
        ...

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        plt.clf()

        a=self.classifier.feature_importances_
        #print(a)
        map = {}
        for i,x in enumerate(a):
            map[x]=i
        od = OrderedDict(sorted(map.items(),reverse=True))
        plt.xlabel('Feature index')
        x=list(od.keys())[:10]
        plt.ylabel('Feature importante')
        y=list(od.values())[:10]
        print(x)
        print(y)
        plt.bar(range(10),x, tick_label=y)
        plt.savefig("2_2_1.png")



def _plot_oob_error():
    plt.clf()

    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    X_train, X_test, t_train, t_test =\
        train_test_split(
            cancer.data, cancer.target,
            test_size=0.3, random_state=109)
    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train,t_train)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig('2_4_1.png')


def _plot_extreme_oob_error():
    plt.clf()
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                bootstrap=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                bootstrap=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    cancer = load_breast_cancer()
    X_train, X_test, t_train, t_test =\
        train_test_split(
            cancer.data, cancer.target,
            test_size=0.3, random_state=109)
    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train,t_train)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig('3_2_1.png')


_plot_extreme_oob_error()
print("Decision Tree Classifier")
classifier_type = sklearn.tree.DecisionTreeClassifier()
cc = CancerClassifier(classifier_type)
print("Accuracy",cc.accuracy())
print("Confusion matrix:\n",cc.confusion_matrix())
print("Precision:",cc.precision())
print("Recall:",cc.recall())
print("Cross validation accuracy:",cc.cross_validation_accuracy())
print("TEST 1.1")

print("Random Forest Classifier")
classifier_type = sklearn.ensemble.RandomForestClassifier(3)
cc = CancerClassifier(classifier_type)
print("Accuracy gotted:",cc.accuracy())
print("Confusion matrix:\n",cc.confusion_matrix())
print("Precision:",cc.precision())
print("Recall:",cc.recall())
print("Cross validation accuracy:",cc.cross_validation_accuracy())
cc.feature_importance()
_plot_oob_error()

print("Extra Tree Classifier")
classifier_type = sklearn.tree.ExtraTreeClassifier()
cc = CancerClassifier(classifier_type)
print("Accuracy gotted:",cc.accuracy())
print("Confusion matrix:\n",cc.confusion_matrix())
print("Precision:",cc.precision())
print("Recall:",cc.recall())
print("Cross validation accuracy:",cc.cross_validation_accuracy())
_plot_extreme_oob_error()



