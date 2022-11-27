def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Investigate The Data
income_data = pd.read_csv("income.csv", header=0, delimiter=", ")
print(income_data.info())
print("\n____________________\n")
print(income_data.iloc[0])
print("\n____________________\n")

# Data Preprocessing
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
# print(income_data["native-country"].value_counts())
"""
Since the majority of the data comes from "United-States", it might make sense to make a column where every row that contains "United-States" becomes a 0 and any other country becomes a 1.
"""
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

labels = income_data[["income"]]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

# Train-Test_Split for training and testing phase
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

# Create Models
forest = RandomForestClassifier(random_state=1)
dtree = DecisionTreeClassifier(random_state=1)

# Train models
forest.fit(train_data, train_labels)
dtree.fit(train_data, train_labels)

# Test models
score1 = forest.score(test_data, test_labels)
score2 = dtree.score(test_data, test_labels)

print("Testing Random Forest score = {}".format(score1))
print("Testing Decision Tree score = {}".format(score2))