import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('train.csv')
test_passengers = pd.read_csv('test.csv')

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':'1', 'male':'0'})
test_passengers['Sex'] = test_passengers['Sex'].map({'female':'1', 'male':'0'})

# Fill the nan values in the age column
replace_age = passengers.mean(axis=1, skipna=True)
passengers['Age'].fillna(value = replace_age, inplace=True)
replace_age = test_passengers.mean(axis=1, skipna=True)
test_passengers['Age'].fillna(value = replace_age, inplace=True)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 1 else 0)
test_passengers['FirstClass'] = test_passengers['Pclass'].apply(lambda p: 1 if p == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 2 else 0)
test_passengers['SecondClass'] = test_passengers['Pclass'].apply(lambda p: 1 if p == 2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
test_features = test_passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size=0.2,)

# Scale the feature data so it has mean = 0 and standard deviation = 1
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Create and train the model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Score the model on the train data
print(lr.score(x_train, y_train))

# Score the model on the test data
print(lr.score(x_test, y_test))

# Analyze the coefficients
print(lr.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
test = np.array([0.5,15.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, test])

# Scale the sample passenger features
sample_passengers = sc.transform(sample_passengers)
test_passengers = sc.transform(test_features)

# Make survival predictions!
print(lr.predict(sample_passengers))
print(lr.predict(test_passengers))