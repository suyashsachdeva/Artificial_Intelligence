import cv2
import numpy as np

# This is the part where we iniiallize our classes 
# And number generator so that we can reproduce our result 
label = ["dog", "cat", "panda"]
np.random.seed(1)

# Generating random values for the weights and the bias
W = np.random.randn(3,3072)
b = np.random.rand(3)


ori =cv2.imread()
image = cv2.resize(ori,(32,32)).flatten()

score = W.dot(image)

for (lab, score) in zip(label, score):
    print("Scores are {} : {:.2f}".format(lab, score))

cv2.putText(ori, "label : {}".format(label[np.argmax(score)]),(10,30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Image", ori)
cv2.waitKey(0)