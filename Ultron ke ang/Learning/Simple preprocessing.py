import cv2

link = r'C:\Users\suyash\Pictures/Camera Roll/imgop.jpg'
img = cv2.imread(link)

class Simple:
	def __init__(self , width, height, inter = cv2.INTER_AREA):

		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		return cv2.resize(img, (200, 200), interpolation = self.inter)

(width, height) = img.shape[:2]
s = Simple(width, height)
cv2.imshow("F",img)
cv2.imshow("off", s.preprocess(img))



cv2.waitKey(0)