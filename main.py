import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

'''
Utility Functions
'''


# Returns a list of the dataset features
def get_features(data, n, start=0):
    return np.array([x[start:n] for x in data if len(x) == len(data[0]) and x[len(x) - 1] != 0])


# Returns a list of the dataset targets
def get_targets(data):
    return np.array([x[len(x) - 1] for x in data if len(x) == len(data[0]) and x[len(x) - 1] != 0])


'''
Cross Validating Functions
'''


def cross_validate_baseline_random_score(features, targets):
    scores = cross_val_score(DummyClassifier(strategy='uniform'), features, targets, cv=5, scoring='accuracy')
    return {'mean': scores.mean(), 'std': scores.std()}


def cross_validate_baseline_most_frequent_score(features, targets):
    scores = cross_val_score(DummyClassifier(strategy='most_frequent'), features, targets, cv=5, scoring='accuracy')
    return {'mean': scores.mean(), 'std': scores.std()}


def cross_validate_logistic_regression(features, targets):
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    means, stds, models = [], [], []

    for C in C_range:
        scores = cross_val_score(LogisticRegression(C=C, penalty='l2', solver='lbfgs'), features, targets, cv=5,
                                 scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())

    br = cross_validate_baseline_random_score(features, targets)
    bf = cross_validate_baseline_most_frequent_score(features, targets)

    plt.figure(figsize=(10, 10))
    plt.errorbar(C_range, [br['mean']] * len(C_range), yerr=[br['std']] * len(C_range), fmt='y')
    plt.errorbar(C_range, [bf['mean']] * len(C_range), yerr=[bf['std']] * len(C_range), fmt='r')
    plt.errorbar(C_range, means, yerr=stds, fmt='b')
    plt.ylabel('Mean Accuracy Score')
    plt.xlabel('C')
    plt.xscale('log')
    plt.title('Comparing accuracy of baseline models with Logistic Regression model')
    plt.legend(['Baseline (random)', 'Baseline (most frequent)', 'Trained model'])
    plt.show()

    print('Mean Accuracy Scores for Logistic Regression')
    print(means)


def cross_validate_linear_svc(features, targets):
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    means, stds, models = [], [], []

    for C in C_range:
        scores = cross_val_score(LinearSVC(C=C), features, targets, cv=5, scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())

    br = cross_validate_baseline_random_score(features, targets)
    bf = cross_validate_baseline_most_frequent_score(features, targets)

    plt.figure(figsize=(10, 10))
    plt.errorbar(C_range, [br['mean']] * len(C_range), yerr=[br['std']] * len(C_range), fmt='y')
    plt.errorbar(C_range, [bf['mean']] * len(C_range), yerr=[bf['std']] * len(C_range), fmt='r')
    plt.errorbar(C_range, means, yerr=stds, fmt='b')
    plt.ylabel('Mean Accuracy Score')
    plt.xlabel('C')
    plt.xscale('log')
    plt.title('Comparing accuracy of baseline models with Linear SVC model')
    plt.legend(['Baseline (random)', 'Baseline (most frequent)', 'Trained model'])
    plt.show()

    print('Mean Accuracy Scores for Linear SVC')
    print(means)


'''
Model Training Functions
'''


def confusion_matrix_random(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = DummyClassifier(strategy='uniform').fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('Confusion Matrix for Baseline Classifier (Random)')
    print(confusion_matrix(Y_test, Y_pred))
    return confusion_matrix(Y_test, Y_pred)


def confusion_matrix_most_frequent(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = DummyClassifier(strategy='most_frequent').fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('Confusion Matrix for Baseline Classifier (Most Frequent)')
    print(confusion_matrix(Y_test, Y_pred))
    return confusion_matrix(Y_test, Y_pred)


def logistic_regression(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = LogisticRegression(C=1, penalty='l2', solver='lbfgs').fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('Confusion Matrix for Logistic Regression Model')
    print(confusion_matrix(Y_test, Y_pred))

    cm_rand = confusion_matrix_random(features, targets)
    cm_freq = confusion_matrix_most_frequent(features, targets)

    fpr, tpr, _ = roc_curve(Y_test, model.decision_function(X_test))
    plt.plot(fpr, tpr)
    rand_tpr = cm_rand[1][1] / (cm_rand[1][1] + cm_rand[1][0])
    rand_fpr = cm_rand[0][1] / (cm_rand[0][1] + cm_rand[0][0])
    freq_tpr = cm_freq[1][1] / (cm_freq[1][1] + cm_freq[1][0])
    freq_fpr = cm_freq[0][1] / (cm_freq[0][1] + cm_freq[0][0])
    plt.plot(rand_tpr, rand_fpr, 'rs')
    plt.plot(freq_tpr, freq_fpr, 'gs')
    plt.title('ROC Curve for Logistic Regression Model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(['Trained model', 'Baseline (random)', 'Baseline (most frequent)'])
    plt.show()


def linear_svc(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = LinearSVC(C=0.1).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('Confusion Matrix for Lienar SVC model')
    print(confusion_matrix(Y_test, Y_pred))

    cm_rand = confusion_matrix_random(features, targets)
    cm_freq = confusion_matrix_most_frequent(features, targets)

    fpr, tpr, _ = roc_curve(Y_test, model.decision_function(X_test))
    plt.plot(fpr, tpr)
    rand_tpr = cm_rand[1][1] / (cm_rand[1][1] + cm_rand[1][0])
    rand_fpr = cm_rand[0][1] / (cm_rand[0][1] + cm_rand[0][0])
    freq_tpr = cm_freq[1][1] / (cm_freq[1][1] + cm_freq[1][0])
    freq_fpr = cm_freq[0][1] / (cm_freq[0][1] + cm_freq[0][0])
    plt.plot(rand_tpr, rand_fpr, 'rs')
    plt.plot(freq_tpr, freq_fpr, 'gs')
    plt.title('ROC Curve for Linear SVC Model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(['Trained model', 'Baseline (random)', 'Baseline (most frequent)'])
    plt.show()


'''
Main
'''


def main():
    error = False
    model = ''

    if len(sys.argv) < 2:
        print('You did not specify a model type.')
        error = True
    else:
        model = str(sys.argv[1])

    games = pd.read_csv('dataset/challenger_games/challenger_games_big.euw1.csv').values.tolist()
    X = get_features(games, len(games[0]) - 1)
    Y = get_targets(games)

    if model == 'logistic-regression':
        cross_validate_logistic_regression(X, Y)
        confusion_matrix_random(X, Y)
        confusion_matrix_most_frequent(X, Y)
        logistic_regression(X, Y)
    elif model == 'linear-svc':
        cross_validate_linear_svc(X, Y)
        confusion_matrix_random(X, Y)
        confusion_matrix_most_frequent(X, Y)
        linear_svc(X, Y)
    elif not error:
        print('Specified model is not supported.')
        error = True

    if error:
        print('The current model types are currently supported:')
        print('    - Logistic Regression (python main.py logistic-regression)')
        print('    - Linear SVC          (python main.py linear-svc)')


if __name__ == "__main__":
    main()
