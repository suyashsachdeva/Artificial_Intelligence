import os
import shutil
import random

VALID = r'C:/Users/suyash/Desktop/valid'
TEST = r'C:/Users/suyash/Desktop/test'
TRAIN = r'C:/Users/suyash/Desktop/asl_alphabet_train/asl_alphabet_train'

li = os.listdir(TRAIN)

for ele in li:
    e = random.sample(os.listdir(TRAIN+'/'+ele), 600)
    os.mkdir(VALID+'/'+ele)
    for v in e:
        os.rename(TRAIN+'/'+ele+'/'+v, VALID+'/'+v)

for ele in li:
    e = random.sample(os.listdir(TRAIN+'/'+ele), 400)
    for v in e:
        os.rename(TRAIN+'/'+ele+'/'+v, TEST+'/'+v)