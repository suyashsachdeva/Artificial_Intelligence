# KNN
# If we give a point at random to the KNN then the KNN will calculate thedistance with the 'k' closest points and 
# then will find out that which of the labels are present in larger numbers. The value of n depends on the amount 
# data that is given as for this example we are taking 15 as the value of 'k'. It is advised that the value of k
# should be odd as it will help in calculation(As odd number their are less chances to get equal values from 2 classes)
# In this we also have the weights so keeping the weights equal it will give equal importance to all the data points and 
# the distance factor will give greater importance to the points closer to it.

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

loc = r'C:\Users\suyash\Desktop\KACHRA\laohub\Smile in Pain\zzz...zzz\Sklearn\car.data'
data = pd.read_csv(loc)

print(data.head())

x = data[['buying', 'maint', 'safety']].values

# Converting the data into numerical value for training
Le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = Le.fit_transform(x[:, i])
print(x)

label = {
    'uacc' : 0,
    'acc' : 1,
    'good' : 2,
    'vgood': 3
}

y = data['class'].map(label)
y = np.array(y, dtype='uint8')

# Spliting our data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

# Training a KNN model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit(xTrain, yTrain)

# Calculating the accuracy
pred = knn.predict(xTest)
acc = metrics.accuracy_score(yTest, pred)
print(acc)