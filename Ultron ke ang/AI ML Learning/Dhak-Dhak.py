# Necessary libraries
import pandas as pd
import numpy as np

# ML tools 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ML algo
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


link = r'C:\Users\suyash\Desktop\KACHRA\Ultron ki chati aulad\heart.csv'
df = pd.read_csv(link)
data_out = df['output']
df = df.drop(columns='output', axis = 1)
data_lr = df.iloc[:,[0,3,4,7,9]]
data_rfc = df.iloc[:,[1,2,5,6,8,10,11,12]]


train_lrx, test_lrx, train_lry, test_lry = train_test_split(data_lr, data_out, test_size = 0.2, random_state = 0)
train_rfcx, test_rfcx, train_rfcy, test_rfcy = train_test_split(data_rfc, data_out, test_size = 0.2, random_state = 0)

lr = LinearRegression()
rfc = RandomForestClassifier()
lr.fit(train_lrx, train_lry)
rfc.fit(train_rfcx,train_rfcy)

y = lr.predict(test_lrx)
y_rfcp = rfc.predict(test_rfcx)

print(accuracy_score(test_lry,y))
print(accuracy_score( test_rfcy,y_rfcp))


