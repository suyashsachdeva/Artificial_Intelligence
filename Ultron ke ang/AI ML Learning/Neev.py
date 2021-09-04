import numpy as np
import matplotlib.pyplot as plt #graph, plotting
import pandas as pd  

dataset=pd.read_csv('Salary_Data.csv')

X=dataset.iloc[:,:-1].values #input
y=dataset.iloc[:,-1].values  #output  -1 refers to the last element

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Plotting of graph
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("salary vs years of expreesion")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()