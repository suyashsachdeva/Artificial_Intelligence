import tensorflow as tf
import os.path 
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix  
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json, model_from_yaml

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

people = np.array(people)
label = np.array(label)
people, label = shuffle(people, label)

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(people.reshape(-1,1))

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# This is the statement which we are using to train our data and in this shuffle is by default True which is a 
# good thing as we don't want any kind of patterns in our data. I don't what is verbose
# We just have to enter valad_size=0.1 
model.fit(x=scaled, y=label, batch_size=10, epochs=40, validation_split=0.1, shuffle=True, verbose=2)

#############################################################################
######################### Making our own custom data ########################
############################################################################
people = []
label = []
# Making the test dataset 
for i in range(10):
    # The 5% of individual which experienced the side effects 
	people.append(randint(13,65))
	label.append(1)

	people.append(randint(65,100))
	label.append(0)

for i in range(200):
    # The rest of our data  
	people.append(randint(13,65))
	label.append(0)

	people.append(randint(65,100))
	label.append(1)

people = np.array(people)
label = np.array(label)
people, label = shuffle(people, label)

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(people.reshape(-1,1))

# This is our predict function and in this also I dont know the use of the verbose function
predict = model.predict(x=scaled, batch_size=10, verbose=0)

for i,p in zip(predict, people):
    print(p,i)

##################################################################################################
#################################### Confusion Matrix ############################################
##################################################################################################

# Rounding of the predictions 
most = np.argmax(predict, axis=-1)
cm = confusion_matrix(y_true=label, y_pred=most)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plt_label = ['No side effects', 'Faced side effects']
plot_confusion_matrix(cm=cm, classes=plt_label, title='Confusion Matrix')

###############################################################################################
################################## SAVING AND LOADING #########################################
###############################################################################################

loc = r'C:\Users\suyash\Desktop\Delete Me\ANN.h5'
locw = r'C:\Users\suyash\Desktop\Delete Me\ANNw.h5'
if os.path.isfile(loc) is False:
    model.save(loc)

# The problem is that if we only save weights then we  have the weights and we dont have the architucture 
# it should only be used when we know the acrhitecture of the model
if os.path.isfile(locw) is False:
    model.save_weights(locw)

# This model saves:- 
#   => The architecture of the model, allowing to re_create the model
#   => The weights of the model
#   => The training condiguration(loss, optimizer)
#   => The state of the optimizer, allowing to resume training exactly where you left off 

new = load_model(loc)
new.load_weights(locw)
print(new.summary())
print(new.get_weights())
print(new.optimizer)

# Json Method
# model.to_json() 
# This method only saves the architechure of the model and unlike the save method which saves on everything

# JSON method to save the string
json_str = model.to_json()

# YAML method 
yaml_str = model.to_yaml()

x_var = model_from_json(json_str)
y_var = model_from_yaml(yaml_str)

print(x_var, y_var)

# We can use model_from_json to import this saved da
