# -*- coding: utf-8 -*-

import importlib,sys

import os
import time
from sklearn import metrics
import numpy as np
import pickle as pickle

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

#importlib.reload(sys)
#sys.setdefaultencoding('utf8')


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print
        para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    thresh = 0.5
    model_save_file = None
    model_save = {}

    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT','GBDT']
    test_classifiers = ['DT', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'GBDT': gradient_boosting_classifier
                   }


    print('reading training and testing data...')
    # tmp = np.loadtxt("winequality-white.csv", dtype=np.str, delimiter=";")
    # X = tmp[1:, 0:11].astype(np.float)  # 加载数据部分
    # y = tmp[1:, 11].astype(np.float)  # 加载类别标签部分

    tmp = np.loadtxt("week-wti.csv", dtype=np.str, delimiter=",")
    X = tmp[1:, 0:8].astype(np.float)  # 加载数据部分
    y = tmp[1:, 8].astype(np.float)  # 加载类别标签部分

    random_state = check_random_state(0)  # 将样本进行随机排列
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))


    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2)

    scaler = StandardScaler()
  #  train_x = scaler.fit_transform(train_x)  # 使用训练数据的均值方差对训练数据进行标准化
  #  test_x = scaler.transform(test_x)

    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        if classifier == 'NB':
            continue
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))