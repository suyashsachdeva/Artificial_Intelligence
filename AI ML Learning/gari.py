from numpy.lib.arraysetops import unique
import pandas as pd 
import numpy as np
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

link = r'C:\Users\suyash\Desktop\KACHRA\laohub\SmileinPain\zzz...zzz\CarPrice_Assignment.csv'

df = pd.read_csv(link)
nor = Normalizer()

#print(df['carlength'])

df['carvolume'] = (df['carlength']*df['carwidth']*df['carheight']) - (df['enginesize']*1000)
df['avgmpg'] = df['citympg']/2 + df['highwaympg']/2


final = df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'enginesize', 'cylindernumber', 'fuelsystem', 'horsepower','avgmpg' ]].copy()
price = df['price'].copy()

#print(final['carbody'].set_values(8,8,1000))                       # This function id used to set values at a particular location

uni = []
y = 0
col = []
for x in final.columns.tolist():
    
    if len(final[x].unique())<20:
        uni.append(final[x].unique().tolist())
        col.append(final.columns.tolist()[y])
    y = y + 1

#print(uni)
#print(type(col[0]))

u = -1
for x in uni:
    u = u + 1
    v = 0
    for i in x:
        final.loc[final[col[u]] == i,col[u]] =  v   
        v = v + 1

final = np.array(final).tolist()
#print(final)

trainx, testx, trainy, testy = train_test_split(final,price,test_size = 0.2, random_state = 42)

lr = LinearRegression()
lr.fit(trainx,trainy)
y = lr.predict(testx)
error = mean_squared_error(testy,y)
errarr = [error]


for degree in range(5):
    
    deg = make_pipeline(PolynomialFeatures(degree+2),LinearRegression())
    deg.fit(trainx,trainy)
    y = deg.predict(testx)
    error = mean_squared_error(testy,y)
    errarr.append(error)

print(math.sqrt(min(errarr)))
