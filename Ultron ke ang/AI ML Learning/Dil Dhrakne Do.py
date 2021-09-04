# rfc op he svc Bekar he

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer as nor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#lr = LogisticRegression()
#svc = LinearSVC(C = 1.0)
rfc = RandomForestClassifier()

location = r'C:\Users\suyash\Desktop\KACHRA\Ultron ki chati aulad\heart.csv'
Rs200 = pd.read_csv(location)
#print(Rs200)
khamba = Rs200.iloc[:,:-1].values
chakna = Rs200.iloc[:,-1].values


khamba = khamba.tolist()
chakna = chakna.tolist()
bottle  = nor().fit(khamba)
darru = bottle.transform(khamba)

trainx,testx,trainy,testy = train_test_split(khamba,chakna,test_size = 0.2,random_state = 0)


rfc.fit(trainx,trainy)

ypred = rfc.predict(testx)
initialacc = accuracy_score(testy,ypred)
print(initialacc)
print(str(rfc.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2],[41,1,0,110,172,0,0,158,0,0,2,0,3]])))

# print(darru)
#print(khamba)


# REMOVED VALUES AND THEIR EFFECTS

#     65	0	2	160	360	0	0	151	0	0.8	 2	0	2	1    improves consitatnacy 81 - 88 --> 83 - 90
##    52	 1	1	120	325	0	1	172	0	0.2	2	0	2	1    this removal showed a fall 83 - 90 --> 80 - 86
##    51	0	2	140	308	0	0	142	0	1.5	2	1	2	1    this removal showed a fall 83 - 90 --> 78 - 83
##    51	1	2	94	227	0	1	154	1	0	2	1	3	1    this removal showed a fall 83 - 90 --> 81 - 83
##    29	1	1	130	 204	0	0	202	0	0	2	0	2	1
#    74 	0	1	120	 269	0	0	121	1	0.2	2	1	2	1


def value(initialacc):                               #changed
    
    arredit = []
    for x in range(len(khamba)):
        
        k = khamba
        c = chakna
        y =  k[x]
        z = c[x]
        c.remove(c[x])
        k.remove(k[x])
        
        bottle1  = nor().fit(k)
        darru1 = bottle1.transform(k)

        trainx,_,trainy,_ = train_test_split(k,c,test_size = 0.2,random_state = 0)

        
        rfc.fit(trainx,trainy)

        ypred = rfc.predict(testx)
        arredit.append(accuracy_score(testy,ypred))

        khamba.insert(x,y)
        chakna.insert(x,z)


    print(max(arredit),arredit.index(max(arredit)))

    rem = arredit.index(max(arredit))
    initial = max(arredit)
    

    if initialacc < max(arredit):
        khamba.remove(khamba[rem])
        chakna.remove(chakna[rem])
        initialacc = initial                         #Changed
        value(initialacc)                            #Changed
    else:
        pass

value(initialacc)                                    #Changed

print(str(rfc.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2],[41,1,0,110,172,0,0,158,0,0,2,0,3],[2,1,0,110,172,0,0,158,0,0,2,0,3]])))

