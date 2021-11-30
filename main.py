import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyRegressor

##
## --------------------------------
##        UTLITY FUNCTIONS
##---------------------------------
##

# Will return data from a specified column
def getDataFromColumn(data, column):
    return [i[column] for i in data]

# Will return a column_stack of the amount of features specified
def getFeatures(data, n):
    ## intialise with first feature from first column
    X = getDataFromColumn(data, 0)
    ## Create Features Array with specified number of features
    for i in range(n):
        if i > 0:
            temp = getDataFromColumn(data, i)
            X = np.column_stack((X, temp))
    return X

# will return a dictionary with the accuracy and standard deviation of the specified model
def getCrossValAccuracy(model, X, Y, cv):
    scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')
    average_accuracy = scores.mean()
    average_accuracy_std = scores.std()

    return {
        "acc": average_accuracy,
        "std": average_accuracy_std
    }

##
## --------------------------------
##          ML METHODS
##---------------------------------
##

def logRegression(features, targets, poly):
    ## Get Features and Target Values
    X = features
    Y = targets

    ## Check for Polynomial Features
    if poly == 0 or None:
        model = LogisticRegression().fit(X, Y)
    else:
        polynomials = PolynomialFeatures(poly)
        XPoly = polynomials.fit_transform(X)
        model = LogisticRegression().fit(XPoly, Y)
    
    return model
    
## BROKEN NEEDS TO BE FIXED
def lasso(features, targets, poly, C):
    ## Get Features and Target Values
    X = features
    Y = targets

    ## Check for Polynomial Features
    if poly == 0 or None:
        model = Lasso(alpha=1/(2 * C)).fit(X, Y)
    else:
        polynomials = PolynomialFeatures(poly)
        XPoly = polynomials.fit_transform(X)
        model = Lasso(alpha=1/(2 * C)).fit(XPoly, Y)

    return model

## BROKEN NEEDS TO BE FIXED
def ridge(features, targets, poly, C):
    ## Get Features and Target Values
    X = features
    Y = targets
    
    ## Check for Polynomial Features
    if poly == 0 or None:
        model = Ridge(alpha=1/(2 * C)).fit(X, Y)
    else:
        polynomials = PolynomialFeatures(poly)
        XPoly = polynomials.fit_transform(X)
        model = Ridge(alpha=1/(2 * C)).fit(XPoly, Y)

    return model

def kNN(features, targets, neighbours):
    ## Get Features and Target Values
    X = features
    Y = targets
    
    model = KNeighborsClassifier(n_neighbors=neighbours).fit(X, Y)
    return model

## BROKEN NEEDS TO BE FIXED
def dummy(features, targets, type):
    ## Get Features and Target Values
    X = np.array(features).reshape(-1, 1)
    Y = np.array(targets).reshape(-1, 1)

    model = DummyRegressor(strategy=type).fit(X, Y)
    return model


##
## --------------------------------
##         MAIN EXECUTION
##---------------------------------
##

df = pd.read_csv('dataset/challenger_games/challenger_games.euw1.csv').values.tolist()

X = getFeatures(df, len(df[0]) - 2)
Y = getDataFromColumn(df, len(df[0]) - 1)
model = ridge(X, Y, 2, 100)

scores = getCrossValAccuracy(model, X, Y, 10)

print(scores["acc"])