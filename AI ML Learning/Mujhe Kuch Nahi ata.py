# In this program i was trying to use regression and classification algorithm to make a sort of constructive interference between
# but i doubt if this is helpful and i am not able to thing of what to do in trying to assign weights to the two results that i will
# get from the 2 algorithms


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer as nor
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split as t
from sklearn.linear_model import LogisticRegression as lr

link = r'C:\Users\suyash\Desktop\KACHRA\Ultron ki chati aulad\heart.csv'

rs200 = pd.read_csv(link)
print(type(rs200))

labels = rs200[:,-1].copy()
khamba = rs200.iloc[:,:-1].values
darru  = rs200.iloc[:,[0,1,2,3]].values
chakna = rs200.iloc[:,-1].values

print(labels)
print(type(darru))
print(darru)
print(np.shape(khamba))

khamba = khamba.reshape(302,13)

khambaupg = []
for x in range(13):
    khambaupg.append(khamba[:,x].reshape(302,1))
    
#darru = khamba.iloc[:,[0,,3,4,,7,9]]
darru = np.concatenate((khambaupg[0],khambaupg[3],khambaupg[4],khambaupg[7],khambaupg[9]),axis = 1)
bottle = np.concatenate((khambaupg[1],khambaupg[2],khambaupg[5],khambaupg[6],khambaupg[8],khambaupg[10],khambaupg[11],khambaupg[12]),axis = 1)

#print(khamba)

darru = darru.reshape(302,5).tolist()
bottle = bottle.reshape(302,8).tolist()
chakna.reshape(302,1)
chakna = chakna.tolist()


trainrlx,testrlx , trainrly,testrly = t(darru,chakna,test_size = 0.2,random_state = 0)
traincx,testcx , traincy,testcy = t(bottle,chakna, test_size = 0.2 , random_state = 0)
print()
print(np.shape(trainrlx),np.shape(trainrly))

trainrly = np.array(trainrly)
testrly = np.array(testrly)
traincy = np.array(traincy)
testcy = np.array(testcy)

trainrly.reshape(241,1).tolist()
testrly.reshape(61,1).tolist()
traincy.reshape(241,1).tolist()
testcy.reshape(61,1).tolist()

print(np.shape(testrly))

lr1 = lr()
rfc1 = rfc()

lr1.fit(trainrlx, trainrly)
rfc1.fit(traincx, traincy)

lry =  lr1.predict(testrlx)
rfcy = rfc1.predict(testcx)

print(acc(testrly,lry))
print(acc(testcy,rfcy))


#bottle = nor.fit(khamba)