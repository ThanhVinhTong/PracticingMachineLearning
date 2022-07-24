import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

filename = "Linear Regression - Sheet1.csv"
percentage = 1
logfile = "log.txt"

def visualize(x, y, w):
    w_0 = w[0][0]
    w_1 = w[1][0]
    x_0 = np.linspace(1, 300, 2)
    y_0 = w_0 + w_1*x_0
    plt.plot(x, y, 'ro')
    plt.plot(x_0, y_0)
    plt.axis([0, 320, 0, 250])
    plt.title("result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def read_data(filename):
    data = pd.read_csv(filename)
    data_values = data.values
    x = np.array( data_values[:, :-1] )
    y = np.array( data_values[:, -1:] )
    return split_data(percentage, x, y)

def split_data(percentage, X, Y):
    split_percentage = int( percentage * X.shape[0] )
    X_copy = X.copy()
    Y_copy = Y.copy()
    return X_copy[:split_percentage], Y_copy[:split_percentage], X_copy[split_percentage:], Y_copy[split_percentage:]

def find_w(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    xT = x.T
    w = np.linalg.inv( xT.dot(x) ).dot( xT.dot(y) )
    return w

def predict(x, w):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    y_predict = x.dot(w)
    return y_predict

def write_to_file(given_data, predicted_data, price, logfile):
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

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data(filename)
    w = find_w(X_train, Y_train)
    visualize(X_train, Y_train, w)
    write_to_file(X_train, predict(X_train, w), Y_train, logfile)