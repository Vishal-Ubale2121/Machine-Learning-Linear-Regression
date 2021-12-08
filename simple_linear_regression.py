#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vishal Ubale
"""
# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
print("X_train: ", X_train)
print("X_test: ", X_test)
print("y_train: ", y_train)
print("y_test: ", y_test)


# Fitting Simple Linear Regression to the Training set
train_regressor = LinearRegression()
train_regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predict = train_regressor.predict(X_test)
print("y_predict: ", y_predict)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, train_regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.grid()
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
# random_x = [2.5]
# random_y = [34500]
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, train_regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()
