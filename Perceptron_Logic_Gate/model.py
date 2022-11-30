from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Creating "AND", "OR", "XOR" Data & Labels
data_AND = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels_AND = [0, 0, 0, 1]
labels_OR = [0, 1, 1, 1]
labels_XOR = [0, 1, 1, 0]

# Visualizing "AND" Data
plt.scatter([point[0] for point in data_AND], [point[1] for point in data_AND], c=labels_AND)

# Building the Perceptron
classifier = Perceptron(max_iter=1000)
classifier.fit(data_AND, labels_AND)
print("AND Gate Perceptron Score: {}".format(classifier.score(data_AND, labels_AND)))

# Visualizing the Perceptron
print("Perceptron's decision_function: {}".format(classifier.decision_function([[0,0], [1,1], [0.5,0.5]])))
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]

distances_matrix = np.reshape(abs_distances, (100, 100))
plt.pcolormesh(x_values, y_values, distances_matrix)
plt.show()