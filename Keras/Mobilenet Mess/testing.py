import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

def preprocess(file):
    loc = r'C:\Users\suyash\Desktop\Sign'
    img = image.load_img(loc+'\\'+file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_dims)

num2alp = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 
            15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z', 26:'del', 27:'', 28:' '}

loc = r'C:\Users\suyash\Desktop\mob.h5'
model = load_model(loc)

img = preprocess('U110.jpg')

ans = model.predict(img)
li = ans.tolist()
m = li[0].index(max(max(li)))

print(m)