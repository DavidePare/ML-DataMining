# Author:Davide Parente 
# Date: 23/10/2022
# Project: Boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

import tools
from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    x=0
    c=0
    for a in X_full['Age']:
        if np.isnan(a) != True:
            x+=a
            c+=1
    X_full['Age'].fillna(round(x/c),inplace=True)
    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)
    return (X_train, y_train), (X_test, y_test), submission_X

def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf = RandomForestClassifier()
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    '''
    print("Confusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
    '''
    return accuracy_score(t_test,prediction),precision_score(t_test,prediction),recall_score(t_test,prediction)

def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf= GradientBoostingClassifier()#n_estimators=100, learning_rate=1.0,  max_depth=1, random_state=0)
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    '''
    print("Confusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
    '''
    return accuracy_score(t_test,prediction),precision_score(t_test,prediction),recall_score(t_test,prediction)


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    estimators =[]
    depth=[]
    learning= []
    a=1
    b=1
    c=0.001

    for i in range(0,99):
        estimators.append(a)
        a+=1
    for i in range(0,49):
        depth.append(b)
        b+=1
    while(c<1):
        learning.append(c)
        if c==0.001:
            c+=0.004
        else:
            c+=0.005
    learning.append(1)
    gb_param_grid = {
        'n_estimators': estimators,
        'max_depth': depth,
        'learning_rate': learning}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    try:
        gb_random = RandomizedSearchCV(
            param_distributions=gb_param_grid,
            estimator=gb,
            scoring="accuracy",
            verbose=0,
            n_iter=50,
            cv=4)
        # Fit randomized_mse to the data
        gb_random.fit(X, y)
    except UserWarning :
        ...
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    clf= GradientBoostingClassifier(n_estimators=69, learning_rate=0.31,  max_depth=2)
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    '''
    print("Confusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
    '''
    return accuracy_score(t_test,prediction),precision_score(t_test,prediction),recall_score(t_test,prediction)

def gb_optimized_train_test2(X_train, t_train, X_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    clf= GradientBoostingClassifier(n_estimators=69, learning_rate=0.31,  max_depth=2)
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    return prediction

def _create_submission():
    '''Create your kaggle submission
    '''
    test = pd.read_csv('./data/test.csv')
    train =pd.read_csv('./data/train.csv')
    featuresTrain= ["Pclass", "Sex", "SibSp", "Parch"]
    y_train = train["Survived"]
    X_train = pd.get_dummies(train[featuresTrain])
    X_test = pd.get_dummies(test[featuresTrain])
    prediction = gb_optimized_train_test2(X_train,y_train,X_test)
    build_kaggle_submission(prediction)



def best_titanic():
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    x=0
    c=0
    for a in X_full['Age']:
        if np.isnan(a) != True:
            x+=a
            c+=1
    X_full['Age'].fillna(round(x/c),inplace=True)

    X_full.drop(
        ['PassengerId','Name'],
        inplace=True, axis=1)

    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    X_full['Embarked'].fillna('S', inplace=True)

    X_dummies = pd.get_dummies(
        X_full,
        columns=['Age','Sex', 'Cabin','Ticket', 'Embarked'],
        drop_first=True)

    X = X_dummies[:len(train)]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)
    return (X_train, y_train), (X_test, y_test)

def _indep():
    (X_train, t_train), (X_test, t_test) =best_titanic()
    clf=RandomForestClassifier()
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    print("RandomForestClassifier\nConfusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
    clf=AdaBoostClassifier()               
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    print("AdaBoostClassifier\nConfusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
    clf= GradientBoostingClassifier()
    clf.fit(X_train,t_train)
    prediction=clf.predict(X_test)
    print("GradientBoostingClassifier\nConfusion matrix:\n",confusion_matrix(t_test,prediction))
    print("Accuracy score:",accuracy_score(t_test,prediction))
    print("Recall score:",recall_score(t_test,prediction))
    print("Precision score:",precision_score(t_test,prediction))
                   


