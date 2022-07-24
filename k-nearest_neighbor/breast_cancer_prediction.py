import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2)

maxx = 0
maxi = 0
k_list = []
accuracies = []

for i in range(100):
  k_list.append(i+1)
  classifier = KNeighborsClassifier(n_neighbors = i+1)
  classifier.fit(training_data, training_labels)
  temp = classifier.score(validation_data, validation_labels)
  accuracies.append(temp)
  if maxx < temp:
    maxx = temp
    maxi = i+1

print("maximum accuracy with k={} is: {}".format(maxi, maxx))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()