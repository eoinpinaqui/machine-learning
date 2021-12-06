import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, \
    roc_curve

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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


def cross_validate_baseline_random_score(features, targets, scoring='accuracy'):
    scores = cross_val_score(DummyClassifier(strategy='uniform'), features, targets, cv=5, scoring=scoring)
    return {'mean': scores.mean(), 'std': scores.std()}


def cross_validate_baseline_most_frequent_score(features, targets, scoring='accuracy'):
    scores = cross_val_score(DummyClassifier(strategy='most_frequent'), features, targets, cv=5, scoring=scoring)
    return {'mean': scores.mean(), 'std': scores.std()}


def cross_validate_logistic_regression(features, targets, mins, max_iter=1000):
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    means, stds, models = [], [], []

    for C in C_range:
        scores = cross_val_score(LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=max_iter), features,
                                 targets, cv=5, scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())

    br = cross_validate_baseline_random_score(features, targets)
    bf = cross_validate_baseline_most_frequent_score(features, targets)

    print('Mean Accuracy Scores for Logistic Regression')
    print(means)

    plt.figure(figsize=(10, 10))
    plt.errorbar(C_range, [br['mean']] * len(C_range), yerr=[br['std']] * len(C_range), fmt='y')
    plt.errorbar(C_range, [bf['mean']] * len(C_range), yerr=[bf['std']] * len(C_range), fmt='r')
    plt.errorbar(C_range, means, yerr=stds, fmt='b')
    plt.ylabel('Mean Accuracy Score')
    plt.xlabel('C')
    plt.xscale('log')
    plt.title('Comparing accuracy of baseline models with Logistic Regression model (' + str(
        mins) + ' minutes dataset)')
    plt.legend(['Baseline (random)', 'Baseline (most frequent)', 'Trained model'])
    plt.show()


def cross_validate_knn(features, targets, mins):
    k_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    means, stds = [], []

    for k in k_range:
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), features, targets, cv=5,
                                 scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())

    br = cross_validate_baseline_random_score(features, targets)
    bf = cross_validate_baseline_most_frequent_score(features, targets)

    plt.figure(figsize=(10, 10))
    plt.errorbar(k_range, [br['mean']] * len(k_range), yerr=[br['std']] * len(k_range), fmt='y')
    plt.errorbar(k_range, [bf['mean']] * len(k_range), yerr=[bf['std']] * len(k_range), fmt='r')
    plt.errorbar(k_range, means, yerr=stds, fmt='b')
    plt.ylabel('Mean Accuracy Score')
    plt.xlabel('k')
    plt.xscale('log', base=2)
    plt.title('Comparing accuracy of baseline models with k Nearest Neighbour model (' + str(
        mins) + ' minutes dataset)')
    plt.legend(['Baseline (random)', 'Baseline (most frequent)', 'Trained model'])
    plt.show()

    print('Mean Accuracy Scores for k Nearest Neighbor')
    print(means)


'''
Model Training Functions
'''


def confusion_matrix_random(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = DummyClassifier(strategy='uniform').fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    print('===== BASELINE (RANDOM) =====')
    print('===== TRAINING DATA =====')
    print('Baseline Model (random) - Accuracy Score: ' + str(accuracy_score(Y_train, Y_pred_train)))
    print('Baseline Model (random) - Precision Score: ' + str(precision_score(Y_train, Y_pred_train)))
    print('Baseline Model (random) - Recall Score: ' + str(recall_score(Y_train, Y_pred_train)))
    print('Baseline Model (random) - F1 Score: ' + str(f1_score(Y_train, Y_pred_train)))

    print('===== TEST DATA =====')
    print('Baseline Model (random) - Accuracy Score: ' + str(accuracy_score(Y_test, Y_pred_test)))
    print('Baseline Model (random) - Precision Score: ' + str(precision_score(Y_test, Y_pred_test)))
    print('Baseline Model (random) - Recall Score: ' + str(recall_score(Y_test, Y_pred_test)))
    print('Baseline Model (random) - F1 Score: ' + str(f1_score(Y_test, Y_pred_test)))
    print('Baseline Model (random) - AUC Score: ' + str(roc_auc_score(Y_test, Y_pred_test)))
    print('Confusion Matrix for Baseline Classifier (most frequent)')
    print('Confusion Matrix for Baseline Classifier (Random)')
    print(confusion_matrix(Y_test, Y_pred_test))
    return confusion_matrix(Y_test, Y_pred_test)


def confusion_matrix_most_frequent(features, targets):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = DummyClassifier(strategy='most_frequent').fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    '''
    print('===== BASELINE (MOST FREQUENT) =====')
    print('===== TRAINING DATA =====')
    print('Baseline Model (most frequent) - Accuracy Score: ' + str(accuracy_score(Y_train, Y_pred_train)))
    print('Baseline Model (most frequent) - Precision Score: ' + str(precision_score(Y_train, Y_pred_train)))
    print('Baseline Model (most frequent) - Recall Score: ' + str(recall_score(Y_train, Y_pred_train)))
    print('Baseline Model (most frequent) - F1 Score: ' + str(f1_score(Y_train, Y_pred_train)))

    print('===== TEST DATA =====')
    print('Baseline Model (most frequent) - Accuracy Score: ' + str(accuracy_score(Y_test, Y_pred_test)))
    print('Baseline Model (most frequent) - Precision Score: ' + str(precision_score(Y_test, Y_pred_test)))
    print('Baseline Model (most frequent) - Recall Score: ' + str(recall_score(Y_test, Y_pred_test)))
    print('Baseline Model (most frequent) - F1 Score: ' + str(f1_score(Y_test, Y_pred_test)))
    print('Baseline Model (most frequent) - AUC Score: ' + str(roc_auc_score(Y_test, Y_pred_test)))
    '''
    print('Confusion Matrix for Baseline Classifier (most frequent)')
    print(confusion_matrix(Y_test, Y_pred_test))
    return confusion_matrix(Y_test, Y_pred_test)


def logistic_regression(features, targets, theta_labels, mins, max_iter=1000):
    if mins == 10:
        C = 1
    else:
        C =0.01

    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=max_iter).fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    model_params = model.coef_[0]

    plt.barh(theta_labels, model_params)
    plt.title('Parameter Weights for Logistic Regression Model (' + str(mins) + ' minutes dataset)')
    plt.ylabel('Parameters')
    plt.xlabel('Weight')
    plt.show()

    cm_rand = confusion_matrix_random(features, targets)
    cm_freq = confusion_matrix_most_frequent(features, targets)

    fpr, tpr, _ = roc_curve(Y_test, model.decision_function(X_test))

    print('===== LOGISTIC REGRESSION =====')
    print('===== TRAINING DATA =====')
    print('Logistic Regression Model - Accuracy Score: ' + str(accuracy_score(Y_train, Y_pred_train)))
    print('Logistic Regression Model - Precision Score: ' + str(precision_score(Y_train, Y_pred_train)))
    print('Logistic Regression Model - Recall Score: ' + str(recall_score(Y_train, Y_pred_train)))
    print('Logistic Regression Model - F1 Score: ' + str(f1_score(Y_train, Y_pred_train)))

    print('===== TEST DATA =====')
    print('Logistic Regression Model - Accuracy Score: ' + str(accuracy_score(Y_test, Y_pred_test)))
    print('Logistic Regression Model - Precision Score: ' + str(precision_score(Y_test, Y_pred_test)))
    print('Logistic Regression Model - Recall Score: ' + str(recall_score(Y_test, Y_pred_test)))
    print('Logistic Regression Model - F1 Score: ' + str(f1_score(Y_test, Y_pred_test)))
    print('Logistic Regression Model - AUC Score: ' + str(roc_auc_score(Y_test, Y_pred_test)))
    print('Confusion Matrix for Logistic Regression Model (' + str(mins) + ' minutes dataset)')
    print(confusion_matrix(Y_test, Y_pred_test))

    plt.plot(fpr, tpr)
    rand_tpr = cm_rand[1][1] / (cm_rand[1][1] + cm_rand[1][0])
    rand_fpr = cm_rand[0][1] / (cm_rand[0][1] + cm_rand[0][0])
    freq_tpr = cm_freq[1][1] / (cm_freq[1][1] + cm_freq[1][0])
    freq_fpr = cm_freq[0][1] / (cm_freq[0][1] + cm_freq[0][0])
    plt.plot(rand_tpr, rand_fpr, 'rs')
    plt.plot(freq_tpr, freq_fpr, 'gs')
    plt.title('ROC Curve for Logistic Regression Model (' + str(mins) + ' minutes dataset)')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(['Trained model', 'Baseline (random)', 'Baseline (most frequent)'])
    plt.show()


def knn(features, targets, mins):
    X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=64).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_pred_t = model.predict(X_train)
    print('Confusion Matrix for k Nearest Neighbour model')
    print(confusion_matrix(Y_test, Y_pred))


    print('===== kNN =====')
    print('===== TRAINING DATA =====')
    print('kNN Model - Accuracy Score: ' + str(accuracy_score(Y_train, Y_pred_t)))
    print('kNN Model - Precision Score: ' + str(precision_score(Y_train, Y_pred_t)))
    print('kNN Model - Recall Score: ' + str(recall_score(Y_train, Y_pred_t)))
    print('kNN Model - F1 Score: ' + str(f1_score(Y_train, Y_pred_t)))
    print(confusion_matrix(Y_train, Y_pred_t))

    print('===== TEST DATA =====')
    print('kNN Model - Accuracy Score: ' + str(accuracy_score(Y_test, Y_pred)))
    print('kNN Model - Precision Score: ' + str(precision_score(Y_test, Y_pred)))
    print('kNN Model - Recall Score: ' + str(recall_score(Y_test, Y_pred)))
    print('kNN Model - F1 Score: ' + str(f1_score(Y_test, Y_pred)))
    print('kNN - AUC Score: ' + str(roc_auc_score(Y_test, Y_pred)))
    print('Confusion Matrix for kNN Model (' + str(mins) + ' minutes dataset)')
    print(confusion_matrix(Y_test, Y_pred))

    cm_rand = confusion_matrix_random(features, targets)
    cm_freq = confusion_matrix_most_frequent(features, targets)

    y_scores = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(Y_test, y_scores[:, 1])

    plt.plot(fpr, tpr)
    rand_tpr = cm_rand[1][1] / (cm_rand[1][1] + cm_rand[1][0])
    rand_fpr = cm_rand[0][1] / (cm_rand[0][1] + cm_rand[0][0])
    freq_tpr = cm_freq[1][1] / (cm_freq[1][1] + cm_freq[1][0])
    freq_fpr = cm_freq[0][1] / (cm_freq[0][1] + cm_freq[0][0])
    plt.plot(rand_tpr, rand_fpr, 'rs')
    plt.plot(freq_tpr, freq_fpr, 'gs')
    plt.title('ROC Curve for k Nearest Neighbour Model')
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

    for i in range(1, 5):
        g = pd.read_csv('dataset/challenger_games_timeline_3/challenger_games_timeline_' + str(i) + '.euw1.csv')
        f = list(g.columns)
        feature_names = f[0:(len(f) - 1)]
        games = g.values.tolist()

        X = get_features(games, len(games[0]) - 1)
        Y = get_targets(games)

        if model == 'logistic-regression':
            mins = i * 5
            cross_validate_logistic_regression(X, Y, mins)
            logistic_regression(X, Y, feature_names, mins)
        elif model == 'knn':
            mins = i * 5
            cross_validate_knn(X, Y, mins)
            knn(X, Y, mins)
        elif not error:
            print('Specified model is not supported.')
            error = True

    if error:
        print('The current model types are currently supported:')
        print('    - Logistic Regression (python main.py logistic-regression)')
        print('    - K Nearest Neighbour (python main.py knn)')


if __name__ == "__main__":
    main()
