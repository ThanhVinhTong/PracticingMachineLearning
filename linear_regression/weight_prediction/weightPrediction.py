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

'''
    y1 = w_1*155 + w_0
    y2 = w_1*160 + w_0

    print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
    print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )
'''

def find_w_formula(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis = 1)
    xT = x.T
    w = np.linalg.inv( xT.dot(x) ).dot(xT.dot(y))
    return w

def find_w_sklearn(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis = 1)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(x, y)
    return regr

if __name__ == '__main__':
    x, y = read_data(filename)
    w = find_w_formula(x, y)
    visualize(x, y, w) 
    regr = find_w_sklearn(x, y)

    # Compare two results
    print( 'Solution found by scikit-learn  : ', regr.coef_ )
    print( 'Solution found by formula: ', w.T)
