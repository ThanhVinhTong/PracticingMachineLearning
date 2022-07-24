import numpy as np
import matplotlib.pyplot as plt
import math

from math import *

"""
    de bai: x = argmin_x(f(x)) = argmin_x(x^2 +5sin(x))
            f'(x) = 2x +5cos(x)
    x_(t+1) = x_t - alpha*(2x_t + 5cos(x))
"""

# tinh dao ham f'(x)
def grad(x):
    return 2*x +5*cos(x)

# tinh gia tri cua ham so f(x)
def cost(x):
    return x**2 + 5*np.sin(x)

def myGD(alpha, x0, bias, stop_point):
    x = [x0]
    for i in range(bias):
        x_new = x[-1] - alpha*grad(x[-1])
        if abs(grad(x_new)) < stop_point:
            break
        x.append(x_new)
    return (x, i)

def execute(alpha, bias, stop_point):
    (x1, i1) = myGD(alpha, -5, bias, stop_point)
    (x2, i2) = myGD(alpha, 5, bias, stop_point)
    
    print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), i1))
    print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), i2))

execute(.1, 100, 1e-3)