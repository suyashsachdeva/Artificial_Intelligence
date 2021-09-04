import numpy as np
import cv2
import os

class LoadingData:
    def __init__(self, preprocessing=None):
        self.preprocessing = preprocessing

        if self.preprocessing is None:
            self.preprocessing = []
    def load(self, imagePaths, verbose=-1):
        # intialize the list of features and labels
        data = []
        labels = []

        for (i, imgPth) in enumerate(imagePaths):
            image = cv2.imread(imgPth)
            label = imgPth.split(os.path.sep)[-2]
            if self.preprocessing is not None:
                for p in self.preprocessing:
                    image = p.preprocessing(image)
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1)%verbose==0:
                print("processed {}/{}".format(i+1, len(imagePaths)))
        return (np.array(data), np.array(labels))
