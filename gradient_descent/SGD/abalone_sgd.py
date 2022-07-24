from time import sleep
from tkinter import W
import numpy as np
import pandas as pd

filename = 'abalone.csv'

class ProcessingData:
    def __init__(self, filename):
        self.filename = filename

    def read_data(self):
        return pd.read_csv(self.filename)

    def preprocess_data(self, data):
        data[data['Type'] == "F"] = 1
        data[data['Type'] == "M"] = 2
        data[data['Type'] == "I"] = 3
        data = data.values
        data = self.normalize_data(data)
        return data

    def train_test_split(self, data, percent, randomState):
        if randomState == 1:
            np.random.shuffle(data)

        n = data[:, 0].size
        a = 0
        b = int((1-percent)*n)
        train = data[a:b, :]
        test = data[b:n, :]
        
        return train, test

    def normalize_data(self, data):
        for i in range(9):
            for j in range(data.shape[0]):
                data[j, i] = np.round_(data[j, i]/max(data[:, i]), 2)
        return data
    
    def process_data(self):
        data = self.read_data()
        data = self.preprocess_data(data)
        train, test = self.train_test_split(data, percent=1/6, randomState=0)
        train, valid = self.train_test_split(train, percent=1/6, randomState=0)
        return train, valid, test

class GradientDescent:
    def __init__(self, train, valid, epochs, learnRate, convergePoint):
        self.train = train
        self.valid = valid
        self.epochs = epochs
        self.learnRate = learnRate
        self.convergePoint = convergePoint

    def loss_calculate(self, yi, yPredict):
        return np.round_(np.sum((yPredict - yi) ** 2), 2)

    def single_grad(self, xi, yi, w):
        return (xi*(xi.dot(w) - yi)).reshape(-1, 1)

    def stochastic_gradient_descent(self):
        D = self.train
        w = np.zeros(D.shape[1]).reshape(-1, 1)
        w_last_check = w
        counter = 0
        one = np.ones((D.shape[0], 1))

        for e in range(self.epochs):
            np.random.shuffle(D)
            x = np.concatenate( (one, D[:, :-1]) , axis=1)
            y = D[:, -1]
            for i in range(D.shape[0]):
                counter += 1
                xi = x[i]
                yi = y[i]

                wDelta = self.single_grad(xi, yi, w)
                w = w - self.learnRate*wDelta

                if counter == 10:
                    counter=0
                    w_this_check = w                 
                    if np.linalg.norm(w_this_check - w_last_check) < 1e-3:                                    
                        return w
                    w_last_check = w_this_check
        return w

class Prediction:
    def __init__(self, w, test):
        self.w = w
        self.test = test

    def predict(self):
        one = np.ones((self.test.shape[0], 1))
        x = np.concatenate( (one, self.test[:, :-1]) , axis=1)
        y = self.test[:, -1]
        print(self.test.shape, x.shape, y.shape, one.shape)
        prediction = x.dot(self.w)
        for i in range(prediction.shape[0]):
            print("Prediction: {} vs Ideal: {}".format(prediction[i], y[i]))

class Score:
    def __init__(self, data, type, weight):
        self.data = data
        self.type = type
        self.weight = weight

    def mae_score(self, x, y, w):
        return np.sum(np.sum(np.abs(x.dot(w) - y), axis = 1), axis = 0) / x.shape[0] #mean absolutely error
    
    def mse_score(self, x, y, w):
        return np.sum(np.sum((x.dot(w) - y)**2, axis = 1), axis = 0) / x.shape[0] #mean squared error

    def score_test(self):
        w = self.weight
        x = self.data[:, :-1]
        y = self.data[:, -1]
        one = np.ones((x.shape[0], 1),)
        x = np.concatenate((one, x), axis=1)

        print("{} MAE score is : {}".format(self.type, round(self.mae_score(x, y, w), 2)))
        print("{} MSE score is : {}".format(self.type, round(self.mse_score(x, y, w), 2)))

class Main:
    def __init__(self, filename):
        self.filename = filename

    def main(self):
        train, valid, test = ProcessingData(self.filename).process_data()
        w = GradientDescent(train, valid, epochs=3000, learnRate=0.01, convergePoint=1e-3).stochastic_gradient_descent()
        Prediction(w, valid).predict()
        Score(test, "test", w).score_test()

Main(filename).main()