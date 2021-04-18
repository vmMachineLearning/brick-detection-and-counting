import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import measure

image = cv2.imread('data/images/pot.jpg')
print("Height: {},Width: {}".format(*image.shape[:2]))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# blurred = cv2.filter2D(gray , -1, kernel)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.bilateralFilter(gray, 21, 41, 41)
# plt.imshow(blurred)

edged = cv2.Canny(blurred, 30, 190, apertureSize=3, L2gradient=True)
# 30, 190 --> pot
# 60, 220 --> guber
plt.imshow(edged)

minLineLength = 100
count = 0
lines = cv2.HoughLinesP(image=edged, rho=1, theta=np.pi/180, threshold=100,
                        lines=np.array([]), minLineLength=minLineLength, maxLineGap=90)

a, b, c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i]
             [0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    count = count + 1
print("count ", count)
cv2.imwrite('houghlines6.jpg', gray)


# Experimental setup
thresh = threshold_local(blurred, 21, offset=15).astype("uint8")*255
mask = cv2.bitwise_not(edged)
thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# plt.imshow(thresh)
thresh_copy = thresh.copy()
(cnts, _) = cv2.findContours(thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     output = cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
# plt.imshow(output)
for c in cnts:
    cv2.drawContours(output, c, -1, (0, 255, 0), 2)
# plt.imshow(output)
# Jump to line

print("# of bricks:", len(cnts))
cv2.imshow("Bricks", output)
cv2.waitKey(0)

num_labels, labels_im = cv2.connectedComponents(thresh, connectivity=4)
labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

for (i, label) in enumerate(np.unique(labels_im)):
    if label == 0:
        continue

    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels_im == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    mask = cv2.add(mask, labelMask)
    cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
print("# of bricks (accurate):", len(np.unique(labels_im)) - 1)
cv2.waitKey(0)
