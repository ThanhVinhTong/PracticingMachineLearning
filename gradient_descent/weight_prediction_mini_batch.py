import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

filename = "fish_weight.csv"

percentage = 2/3
alpha = 0.02
bias = 10
epsilon = 1e-3
batch_size = 10
num_iters = 3000

def name_fish(i):
    switcher={
                'Bream'     :1., 
                'Parkki'    :2., 
                'Perch'     :3., 
                'Pike'      :4., 
                'Roach'     :5., 
                'Smelt'     :6., 
                'Whitefish' :7.
             }
    return switcher.get(i,"Fish's name not exist")

def visualize(list_lost):
    plt.plot(list_lost)
    plt.title("Loss Graph")
    plt.ylabel('loss value')
    plt.xlabel('eplison')
    plt.show()

def predict(filename, a, w):
    a = add_one(a)
    write_to_file(filename, a.dot(w))

def read_data():
    data = pd.read_csv(filename)
    data_value = data.values
    j=0
    for i in data_value[:, 0]:
        data_value[j, 0] = name_fish(i)
        j+=1
    data_value = np.array(data_value, dtype=np.float64)
    #data_value = np.array(feature_normalized(data_value), dtype=np.float64)
    #return np.concatenate((data_value[:, 0:1], data_value[:, 2:]), axis=1), data_value[:, 1:2]
    return data_value[:, 2:], data_value[:, 1:2]

def feature_normalized(feature):
    return (feature - np.min(feature, axis = 0))/(np.max(feature, axis = 0) - np.min(feature, axis = 0))

def split_data(x, y):
    n = int(percentage*x.shape[0])
    return x[:n, :], x[n:, :], y[:n, :], y[n:, :]

def add_one(a):
    one = np.ones((a.shape[0], 1))
    return np.concatenate((one, a), axis=1)

def check_converged(w_old, w_new):
    if np.sum(np.linalg.norm(w_new - w_old)) < epsilon:                                    
        return 1 
    return 0

def MGD(x, y, weight_val):
    list_lost = []
    x = add_one(x)
    w = np.full((x.shape[1], 1), weight_val)
    for i in range(num_iters):
        random = np.random.permutation(int(x.shape[0]/batch_size))
        w_new = None
        sum_lost = 0
        for j in random:
            xj = x[j*batch_size : (j+1)*batch_size]
            yj = y[j*batch_size : (j+1)*batch_size]
            zj = xj.dot(w)
            w_new = w - alpha * (xj.T).dot(zj-yj)
            sum_lost += np.sum(np.sum((zj - yj) ** 2, axis = 1), axis = 0)/batch_size
            if check_converged(w, w_new): 
                break
            else:
                w = w_new
        list_lost.append(sum_lost)
    visualize(list_lost)
    return w

def score_test(X, Y, weight, train_test):
    X = add_one(X)
    mae_score = np.sum(np.sum(np.abs(X.dot(weight) - Y), axis = 1), axis = 0) / X.shape[0] #mean absolutely error
    mse_score = np.sum(np.sum((X.dot(weight) - Y)**2, axis = 1), axis = 0) / X.shape[0] #mean squared error
    print("{} MAE score is : {}".format(train_test, round(mae_score, 2)))
    print("{} MSE score is : {}".format(train_test, round(mse_score, 2)))

def execute():
    x, y = read_data()
    x = feature_normalized(x)
    x_train, x_test, y_train, y_test = split_data(x, y)

    write_to_file("train.txt", x_train)
    write_to_file("realValue.txt", y_test)
    w = MGD(x_train, y_train, 0.002)
    predict("log.txt", x_test, w)
    score_test(x_train, y_train, w, "train")
    score_test(x_test, y_test, w, "test")

def write_to_file(filename, a):
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "a") as o:
        for i in range(len(a)):
            o.write('{} : {}'.format(i, a[i]) )
            #o.write('{}'.format(np.round_(a[i], 2)) )
            o.write('\n')
        o.close()

execute()