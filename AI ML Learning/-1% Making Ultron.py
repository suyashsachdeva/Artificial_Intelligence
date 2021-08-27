#impoting
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  

#file
local = r"C:\Users\suyash\Desktop\KACHRA\AI ML and other project stuff\50_Startups.csv"
dataset=pd.read_csv(local)

#file to desired formate
x=dataset.iloc[:,:-1].values #input
Y=dataset.iloc[:,-1].values  #output  -1 refers to the last element


# Normalize function present in the sklearn library
a=[]
b=[]

#file optimization
for i in x[:,-1]:
    if i=="Florida":
        a.append(-0.5)
        b.append(1.0)
    elif i=="California":
        a.append(0.0)
        b.append(1.0)
    elif i=="New York":
        a.append(0.5)
        b.append(1.0)

ar = np.array(a)
br = np.array(b)
arr = ar.reshape(50,1) 
brr = br.reshape(50,1)               
xx = np.array(x)
xd = np.delete(xx , (3), axis= 1)
Array = np.concatenate((xd,arr), axis= 1)
#print(Array)
YAR = np.array(Y)
y = YAR.reshape(50,1)

arr1, arr2 , arr3, arr4 = np.hsplit(Array, 4)
#print(arr1)
#print(arr3)

arri = (arr1 - np.mean(arr1))/(np.max(arr1) - np.min(arr1))
#print(arri)
arrii = (arr2 - np.mean(arr2))/(np.max(arr2) - np.min(arr2))

arriii = (arr3 - np.mean(arr3))/(np.max(arr3) - np.min(arr3))
array = np.concatenate((arri,arrii,arriii,arr4), axis= 1)
#print(array)

#y= y1 + arr1 + arr2 + arr4
#print(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression 

# Splitting of data
array_train, array_test, y_train, y_test = train_test_split(array ,y , test_size=0.2,random_state=0)

# Linear Regression
regressor = LinearRegression()
regressor.fit(array_train, y_train)
y_pred = regressor.predict(array_test)
arrit, arriit, arriiit, arrt = np.hsplit(array_train, 4)

# Quadratic Regression 
degree = 2
pol = make_pipeline(PolynomialFeatures(degree), LinearRegression())
pol.fit(array_train, y_train)
y_pred = pol.predict(array_test)

# User Regression
degree = int(input("Enter the degre of regression yoou want to try"))
polhalf = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polhalf.fit(array_train, y_train)
y_pred = polhalf.predict(array_test)

# Inputing the values from the user
val1 ,val2, val3, val4 = input("Please enter the\nR and D spending, Administration, Marketing and the city respectivily :").split(",")
if val4=="Florida":
    value4 = -0.5
elif val4=="California":
    value4=0.0
elif val4=="New York":
    value4 = 0.5 
else:
    value4 = 0.0              

#Variables
value1 = (float(val1) - np.mean(arr1))/(np.max(arr1))
value2 = (float(val2) - np.mean(arr2))/(np.max(arr2))
value3 = (float(val3) - np.mean(arr3))/(np.max(arr3))

#output section
print("The value when we take linear regression"+str(regressor.predict([[value1 , value2, value3, value4]])))
print("The value when we take quadratic regression"+str(pol.predict([[value1, value2, value3, value4]])))
print("The value when we take polynomial regression"+str(polhalf.predict([[value1, value2, value3, value4]])))
#|-----x-------------------x---------------------x--------------------x---------------x-------------x---------|#

#Graphical Display program
yesno = str(input("Would you like to see the graphs of the regression equation formed(Y/N)"))
if yesno=="Y" or yesno=="y":
    #1. Plotting the graph for a linear equation
    """
    plt.scatter(arrit, y_train, color='red')
    plt.plot(arrit, regressor.predict(array_train),color='blue')
    plt.title("OMG")
    plt.xlabel("years of experience")
    plt.ylabel("Profit of the company")
    """

    #2. Plotting the graph for a quadratic equation
    plt.scatter(arrit, y_train, color='red')
    plt.plot(arrit, pol.predict(array_train),color='blue')
    plt.title("OMG! This is quad eq. gph")
    plt.xlabel("R & D spending")
    plt.ylabel("Profit of the company")
    plt.show() 


    plt.scatter(arriit, y_train, color='red')
    plt.plot(arriit, pol.predict(array_train),color='blue')
    plt.title("OMG! This is quad eq. gph")
    plt.xlabel("Administrative spending")
    plt.ylabel("Profit of the company")
    plt.show() 


    plt.scatter(arriiit, y_train, color='red')
    plt.plot(arriiit, pol.predict(array_train),color='blue')
    plt.title("OMG! This is quad eq. gph")
    plt.xlabel("Market spending")
    plt.ylabel("Profit of the company")
    plt.show() 


    plt.scatter(arrt, y_train, color='red')
    plt.plot(arrt, pol.predict(array_train),color='blue')
    plt.title("OMG! This is quad eq. gph")
    plt.xlabel("City=> New York,California,Florida = 0.5,0.0,-0.5")
    plt.ylabel("Profit of the company")
    plt.show() 
    print("Goodbye")   
else: 
    print("Goodbye")