import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets, linear_model

filename = "weightdata.csv"

def read_data(filename):
    data = pd.read_csv(filename)
    data_values = data.values
    x = np.array(data_values[:, :-1])
    y = np.array(data_values[:, -1:])
    return x, y

def visualize(x, y, w):
    w_0 = w[0][0]
    w_1 = w[1][0]
    x_0 = np.linspace(145, 185, 2)
    y_0 = w_0 + w_1*x_0
    plt.plot(x, y, 'ro')
    plt.plot(x_0, y_0)
    plt.axis([140, 190, 45, 75])
    plt.title("Checking linear or not")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.show()

def loss_function(x, y, w):
    loss_list = []

    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis = 1)
    temp = x.dot(w)
    temp = temp - y

    print(temp)

def find_w_formula(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis = 1)
    xT = x.T
    w = np.linalg.inv( xT.dot(x) ).dot(xT.dot(y))
    return w

if __name__ == '__main__':
    x, y = read_data(filename)
    w = find_w_formula(x, y)
#    visualize(x, y, w) 
    loss_function(x, y, w)
    print( 'Solution found by formula: ', w.T)
