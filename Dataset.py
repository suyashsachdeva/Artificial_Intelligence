# We should learn multiple apki so that the snall problems that we face don't seem too much
# Even in job interviews they ask you about multiplel libraries and ask you question on the difference between the two apki

# Sequential Model :- 
# Fit:- x and y as the input in the numpy array and x, y should be in the same formate
# Normalization:- Using this depends on the type of dataset and what model are we using to train


# This is the dataset making the process where we  are making a dataset to train the model
from random import randint
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Intialization
people = []
label = []


# Making the dataset 
for i in range(50):
    # The 5% of individual which experienced the side effects 
	people.append(randint(13,65))
	label.append(1)

	people.append(randint(65,100))
	label.append(0)

for i in range(1000):
    # The rest of our data  
	people.append(randint(13,65))
	label.append(0)

	people.append(randint(65,100))
	label.append(1)

# I hate this mujhko koi oproject par kaam karna he kisi aur ke final year assigments par nahi 
# mujhko apne projects karne he na ki kisi aur ke

people = np.array(people)
label = np.array(label)
people, label = shuffle(people, label)

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(people.reshape(-1,1))

for i in scaled:
	print(i)