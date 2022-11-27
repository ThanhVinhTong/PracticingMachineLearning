import numpy as np

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# import dataset
digits = datasets.load_digits()

# Look through the dataset
"""
print(digits.DESCR)
print(digits.data)
print(digits.target)
"""

# visualize data image
"""
plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])
"""

# Creating model and fit model
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualize all centroids created by model
fig = plt.figure(figsize = (8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1+i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

# Testing the model
new_samples = np.array([
[0.00,0.00,0.00,0.00,4.11,5.26,0.00,0.00,0.00,0.00,0.00,1.14,7.55,6.86,0.00,0.00,0.00,0.00,0.00,4.80,7.62,6.86,0.00,0.00,0.00,0.00,2.29,7.55,7.55,6.86,0.00,0.00,0.00,0.00,6.71,6.10,5.03,6.86,0.00,0.00,0.00,0.00,2.13,1.07,4.57,6.86,0.00,0.00,0.00,0.00,0.00,0.00,3.96,5.95,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,5.26,6.86,6.86,6.33,5.26,0.76,0.00,0.00,3.51,3.96,5.11,7.62,6.48,0.99,0.00,0.30,0.76,0.76,4.19,7.62,0.53,0.00,0.00,5.72,7.62,7.62,7.62,7.62,7.55,6.41,0.23,1.29,2.29,2.29,6.48,6.56,3.58,4.12,0.00,0.00,0.00,0.00,6.63,5.11,0.00,0.00,0.00,0.00,0.00,0.00,2.82,1.45,0.00,0.00,0.00],
[0.00,0.00,0.00,0.61,0.30,0.00,0.00,0.00,0.00,0.00,3.13,7.62,5.26,0.00,0.00,0.00,0.00,0.00,6.71,6.64,1.68,0.00,0.00,0.00,0.00,0.00,7.62,7.62,7.63,6.86,5.03,0.69,0.00,0.00,7.40,5.64,3.51,5.26,7.62,3.05,0.00,0.00,5.80,6.86,0.61,5.26,7.47,1.30,0.00,0.00,2.37,7.62,7.40,7.62,4.12,0.00,0.00,0.00,0.00,2.67,4.57,3.20,0.00,0.00],
[0.00,0.08,1.83,2.29,1.60,0.15,0.00,0.00,0.00,3.28,7.62,7.62,7.62,5.95,0.00,0.00,0.00,4.57,7.17,1.30,6.41,7.32,0.23,0.00,0.00,3.74,7.63,6.48,7.62,7.62,0.76,0.00,0.00,0.61,3.97,4.57,5.26,7.62,1.22,0.00,0.00,0.00,1.52,1.68,2.29,7.62,1.53,0.00,0.00,0.00,4.96,7.17,5.95,7.63,1.14,0.00,0.00,0.00,1.45,6.33,6.79,3.20,0.00,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(1, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(3, end='')
  elif new_labels[i] == 4:
    print(4, end='')
  elif new_labels[i] == 5:
    print(5, end='')
  elif new_labels[i] == 6:
    print(6, end='')
  elif new_labels[i] == 7:
    print(7, end='')
  elif new_labels[i] == 8:
    print(8, end='')
  elif new_labels[i] == 9:
    print(9, end='')

print(new_labels)