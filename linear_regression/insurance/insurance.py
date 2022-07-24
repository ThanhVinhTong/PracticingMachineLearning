import numpy as np
from numpy.core.shape_base import hstack
import pandas as pd
import csv  
import os

from sklearn import datasets, linear_model

filename = "insurance.csv"
logfile = "log.txt"
logfiletest = "log2.txt"

def read_data(filename):
    data = pd.read_csv(filename)
    data_values = data.values
    temp = np.array(data_values[:, :1])
    x = np.array(data_values[:, 2:-1])
    x = np.hstack((temp, x))
    y = np.array(data_values[:, -1:])
    return x, y

def split_data(split, X, Y):
    split_percentage = int(split * X.shape[0])
    X_copy = X.copy()
    Y_copy = Y.copy()
    return X_copy[:split_percentage], Y_copy[:split_percentage], X_copy[split_percentage:], Y_copy[split_percentage:]

def digitalize(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] == "no":
                x[i][j] = 1010.10101
            if x[i][j] == "yes":
                x[i][j] = 101.01010
            if x[i][j] == "southwest":
                x[i][j] = 15666.0
            if x[i][j] == "northwest":
                x[i][j] = 17469.0
            if x[i][j] == "southeast":
                x[i][j] = 19398.0
            if x[i][j] == "northeast":
                x[i][j] = 21901.0
    return x

def find_w_formula(x, y):
    x = x.astype('float')
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    xT = x.T
    w = np.linalg.inv( xT.dot(x) ).dot( xT.dot(y) )
    return w

def score_test(X, Y, weight, train_test):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    mae_score = np.sum(np.sum(np.abs(X.dot(weight) - Y), axis = 1), axis = 0) / X.shape[0] #mean absolutely error
    mse_score = np.sum(np.sum((X.dot(weight) - Y)**2, axis = 1), axis = 0) / X.shape[0] #mean squared error
    print("{} MAE score is : {}".format(train_test, mae_score))
    print("{} MSE score is : {}".format(train_test, mse_score))

def predict(x, w):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    y_predict = x.dot(w)
    return y_predict

def write_to_file(given_data, predicted_data, price, logfile):
    predicted_data = np.round(predicted_data.astype(np.float64), 3)
#    given_data = np.round(given_data, 2)
    price = np.round(price.astype(np.float64), 3)
    if os.path.exists(logfile):
        os.remove(logfile)

    with open(logfile, "a") as o:
        o.write('{}\t\t'.format("Predicted Price"))
        o.write('{}\t\t'.format("Real Price"))
        o.write('{}\t\t'.format("Given Data"))
        o.write('\n')
        for i in range(len(given_data)):
            o.write('{}\t\t'.format(predicted_data[i]) )
            o.write('{}\t\t'.format(price[i]) )
            o.write('{}\t\t'.format(given_data[i]) )
            o.write('\n')
        o.close()

def find_w_sklearn(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis = 1)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(x, y)
    return regr

if __name__ == "__main__":
    x,y = read_data(filename)
    x = digitalize(x)
    
    x_train, y_train, x_test, y_test =  split_data(0.75, x, y)
    w = find_w_formula(x_train, y_train)
    regr = find_w_sklearn(x_train, y_train)

    write_to_file(x_test, predict(x_test, w), y_test, logfile)
    write_to_file(x_test, predict(x_test, regr.coef_.T), y_test, logfiletest)

    print("Score for Predicting insurance' dataset:\n")
    score_test(x_train, y_train, w, "Train")
    score_test(x_test, y_test, w, "Test")