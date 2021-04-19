from recognizer import countBricks
import cv2

image = cv2.imread('data/images/test5.jpg')
print("Height: {},Width: {}".format(*image.shape[:2]))

countBricks(image, 60, 250, 8)
