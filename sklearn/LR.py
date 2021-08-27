######################################################################################################3
#### PERSONALLY I HATE THIS PROGRAM AND WOULDN'T TO TALK ABOUT THIS PROGRAM ########################


# Linear regression
import numpy as np 
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

bost = datasets.load_boston()

# Features and labels
x = bost.data
y = bost.target

print(x.shape, y.shape, x[1,])

# Our model to train on the data
lr = linear_model.LinearRegression()

# The graph with the y-axis and one variable of the equation 
plt.scatter(x.T[12], y)
#plt.show()

# Spliting of the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
# Training of the model
model = lr.fit(xtrain, ytrain)
pred = model.predict(xtest)
print("The RMS value of the error", lr.score(x,y))
print("coffercient", lr.coef_)  # Its the slope of all the fit lines that we have
print("Intersept", lr.intercept_)
acc = mean_squared_error(ytest,pred)
print(acc)
