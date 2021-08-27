# Classification 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()

# Spliting features for the target values
x = iris.data
y = iris.target

print(x.shape, y.shape)

# Splilting our data in two parts 80% for training and the rest for testing the data. 
# This is a very standard spliting of data in machine learning
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

