import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import *
from sklearn import datasets, neighbors
from numpy import argsort
from sklearn.metrics import accuracy_score

def init_data():
    iris = datasets.load_iris()
    iris_data = np.asarray(iris.data)
    iris_target = np.asarray(iris.target)
    label = np.unique(iris_target)
    x, y = shuffle_data(iris_data, iris_target)
    return x, y, label

def shuffle_data(a, b):
    temp = np.hstack((a, b.reshape(len(b), 1)))
    np.random.shuffle(temp)
    return temp[:, :a.shape[1]], temp[:, a.shape[1]:].reshape(len(b))

def split_data(x, y):
    nx = int(2*(x.shape[0])/3) #2/1 ratio
    ny = int(2*(y.shape[0])/3)
    return x[:nx, :], x[nx:, :], y[:ny], y[ny:]

def train(x_train, y_train, label):
    D = [()]
    for j in label:
        D.append((x_train[y_train==j ,:], j))
    return D

def minkowski_distance(x, y, p):
    dist = 0
    for a,b in zip(x,y):
        dist = dist + abs(a-b)**p
    dist = dist ** (1/p)
    return round(dist, 3)

def test_result(x_train, x_test, y_train, y_test, y_predict, n):
    print(y_predict)
    print(y_test)
    print("Accuracy of " + str(n) + "NN, own code: " + 
            str(100*accuracy_score(y_test, y_predict)) + "%")
    clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy of " + str(n) + "NN with scikit learn neighbor: " + 
            str(100*accuracy_score(y_test, y_pred)) + "%")

def execute(n, p):
    x, y, label = init_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    D = train(x_train, y_train, label)
    y_predict = []
    for test_point in x_test:
        i=0
        distances = []
        for train_point in x_train:
            dist = minkowski_distance(test_point, train_point, p)
            distances.append((dist, i))
            i+=1
        distances = np.asarray(distances)
        count_labels = []
        for j in distances[:, 0].argsort():
            if len(count_labels) == n:
                break
            count_labels.append(y_train[j])
            
        y_predict.append(np.bincount(count_labels).argmax())

    test_result(x_train, x_test, y_train, y_test, y_predict, n)

execute(10, 2)