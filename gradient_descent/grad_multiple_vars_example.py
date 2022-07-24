from re import X
import numpy as np
import matplotlib.pyplot as plt

def init_data():
    np.random.seed(2)
    x = np.random.rand(1000, 1)
    y = 4 + 3*x + .2*np.random.randn(1000, 1) # noise added
    return x, y

def grad(w, a, b):
    return np.dot(a.T, (np.dot(a, w) - b))/(a.shape[0])

def GD(learn_rate, w, bias, stop_point, a, b):
    w_new = [w]
    for i in range(bias):
        w_new.append(w - learn_rate*grad(w_new[-1], a, b))
        if np.linalg.norm(w_new[-1] - w_new[-2]) < stop_point:
            break
    return (w_new[-1], i)

def linear(x, y, learn_rate, bias, stop_point):
    x_copy = x.copy()
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    w = np.dot( np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))
    w, i = GD(learn_rate, w, bias, stop_point, x, y)

    def test_prediction(w):
        x_predict = np.linspace(0, 1, 2, endpoint=True)
        y_predict = w[0][0] + w[1][0]*x_predict
        cost = loss_func(y_predict, y)
        print(cost)
        visualize(x_copy, y, x_predict, y_predict)

    test_prediction(w)
    print('Solution found by GD: w = ', w.T, ',\nafter %d iterations.' %(i+1))

def loss_func(y_predict, y):
    lost = 0
    lost = np.sum((y_predict - y) ** 2)
    return lost/(2*len(y))

def visualize(x, y, a, b):
    plt.plot(x.T, y.T, 'b.')
    plt.plot(a, b, 'y', linewidth=2)
    plt.axis([0, 1, 0, 10])
    plt.show()

def execute():
    x, y = init_data()
    linear(x, y, 1, 100, 1e-3)

execute()