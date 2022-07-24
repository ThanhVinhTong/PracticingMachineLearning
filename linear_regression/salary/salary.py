import numpy as np
from numpy.core.shape_base import hstack
import pandas as pd
import matplotlib.pyplot as plt

filename = 'salary.csv'

def visualize(x, y, z):
    plt.plot(x.T, y.T, marker='o', ms = 5, mfc = 'green', mec = 'green')
    plt.plot(x.T, z.T, marker = 'o', ms = 5, mfc = 'r', mec = 'red')
    plt.grid()
    plt.show()

def read_data():
    data = pd.read_csv(filename)
    data_values = data.values
    x = np.array(data_values[:, :1])
    y = np.array(data_values[:, -1:])
    return x, y

def split_data(split, x, y):
    percentage = int(split*(x.shape[0]))
    x_copy = x.copy()
    y_copy = y.copy()
    return x_copy[:percentage], y_copy[:percentage], x_copy[percentage:], y_copy[percentage:]

def algorithm(a, b):
    one = np.ones((a.shape[0], 1))
    a = np.concatenate((one, a), axis=1)
    w = np.dot(np.linalg.inv(np.dot(a.T,a)),(np.dot(a.T, b)))
    return w
    
def train():
    x, y = read_data()
    x_train, y_train, x_test, y_test = split_data(0.75, x, y)
    w = algorithm(x_train, y_train)
    return x_test, y_test, w

def predict(x_test, y_test, w):
    one = np.ones((x_test.shape[0], 1))
    temp = np.concatenate((one, x_test), axis=1)
    y_predict = np.dot(temp, w)
    return y_predict

def execute():
    x_test, y_test, w = train()
    y_predict = predict(x_test, y_test, w)
    visualize(x_test, y_test, y_predict)

execute()