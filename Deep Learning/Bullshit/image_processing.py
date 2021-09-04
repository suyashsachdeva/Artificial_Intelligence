import cv2

class Preprocess:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter

    def resizer(self, image):
        return cv2.resize(image, (self.height, self.width), interpolation=self.inter)