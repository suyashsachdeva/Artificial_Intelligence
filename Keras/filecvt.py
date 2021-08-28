import os

OLD = r'C:\Users\suyash\Desktop\asl_alphabet_train\asl_alphabet_train'
NEW = r'C:\Users\suyash\Desktop\Sign'

element = os.listdir(OLD)

for e in element:
    ele = os.listdir(OLD+'/'+e)
    for x in ele:
        NAME = os.path.basename(x)
        os.rename(OLD+'/'+e+'/'+NAME, NEW+'/'+NAME)
