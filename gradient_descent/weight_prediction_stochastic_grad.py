from turtle import color
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import metrics

filename = "weightdata.csv"

percentage = 2/3
alpha = 1
bias = 10
epsilon = 1e-3

def visualize(x, y, w, xmax, ymax):
    x_copy = x.copy()
    one = np.ones((x.shape[0], 1),)
    x_copy = np.concatenate((one, x_copy), axis=1)
    print(w)
    y_pred = x_copy.dot(w)
    print(np.round_(x,3), '\n\n', np.round_(y_pred,3), '\n\n', np.round_(y,3))
    # Drawing the fitting line 
    plt.plot(x, y, marker='o', color = 'red')
    plt.plot(x, y_pred, marker='o' , color='green')    
    plt.xlabel('Height (cm)/{}'.format(xmax))
    plt.ylabel('Weight (kg)/{}'.format(ymax))
    plt.show()

def read_data():
    data = pd.read_csv(filename)
    data_value = data.values
    return data_value[:, :-1], data_value[:, -1:]

def normalize_data(x, y):
    return x/np.amax(x), y/np.amax(y), np.amax(x), np. amax(y)

def denormalize_data(x, y, xtrain, ytrain, xtest, ytest, xmax, ymax):
    return x*xmax, y*ymax, xtrain*xmax, ytrain*ymax, xtest*xmax, ytest*ymax,

def split_data(x, y):
    n = int(percentage*x.shape[0])
    return x[:n], x[n:], y[:n], y[n:]

def linear(a, b):
    one = np.ones((a.shape[0], 1))
    a = np.concatenate((one, a), axis=1)
    VT = np.dot(a.T, a)
    VP = np.dot(a.T, b)
    return np.dot(np.linalg.pinv(VT), VP)

def single_grad(a, b, w, index):
    one = np.ones((a.shape[0], 1),)
    a = np.concatenate((one, a), axis=1)
    xi = a[index, :]
    yi = b[index]
    temp = np.dot(xi, w) - yi
    return ((xi.T)*temp).reshape(2,1)

def check_converged(w_old, w_new):
    if np.linalg.norm(w_new - w_old)/len(w_old) < epsilon:                                    
        return 1 
    return 0

def SGD(x, y, w):
    w_new = w
    n = x.shape[0]
    for i in range(bias):
        # shuffle data
        random_index = np.random.permutation(n)
        for j in random_index:
            g = single_grad(x, y, w_new, j)
            w_old = w_new.copy()
            w_new = w_old - alpha*g
            if check_converged(w_old, w_new) == 1:
                break
    return w_new

def score_test(X, Y, weight, train_test):
    one = np.ones((X.shape[0], 1),)
    X = np.concatenate((one, X), axis=1)
    mae_score = np.sum(np.sum(np.abs(X.dot(weight) - Y), axis = 1), axis = 0) / X.shape[0] #mean absolutely error
    mse_score = np.sum(np.sum((X.dot(weight) - Y)**2, axis = 1), axis = 0) / X.shape[0] #mean squared error
    print("{} MAE score is : {}".format(train_test, round(mae_score, 2)))
    print("{} MSE score is : {}".format(train_test, round(mse_score, 2)))

def execute():
    x, y = read_data()
    x, y, xmax, ymax = normalize_data(x, y)
    x_train, x_test, y_train, y_test = split_data(x, y)
    w = np.round_(linear(x_train, y_train), decimals = 2)
    w = SGD(x_train, y_train, w)
    #x, y, x_train, y_train, x_test, y_test = denormalize_data(x, y, x_train, y_train, x_test, y_test, xmax, ymax)
    score_test(x_train, y_train, w, "train")
    score_test(x_test, y_test, w, "test")
    visualize(x, y, w, xmax, ymax)

execute()