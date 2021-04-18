import numpy as np
import cv2
from skimage.filters import threshold_local
from skimage import measure
import imutils

image = cv2.imread('data/images/test5.jpg')
print("Height: {},Width: {}".format(*image.shape[:2]))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')

blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# plt.imshow(blurred)

# Watch for the argument
#edges = cv2.Canny(gray, 100, 200)

# lap = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)#cv2.Laplacian(blurred,cv2.CV_64F)
edged = cv2.Canny(blurred, 110, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edged, kernel)
cv2.imshow("dilate", dilated)

thresh = threshold_local(dilated, 21, offset=15).astype("uint8")*255
cv2.imshow('thresh', thresh)
mask = cv2.bitwise_not(edged)
thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

(cnts, _) = cv2.findContours(dilated.copy(),
                             cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# output = thresh.copy()
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     output = cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.imshow("something", output)


output = image.copy()
# cot = 0
for c in cnts:
    cv2.drawContours(output, c, -1, (0, 255, 0), 2)
    # cot = cnts.h_next()
# plt.imshow(output)


print("# of bricks:", len(cnts))
# print("# of bricks:", len(cot))

cv2.imshow("Bricks", output)
cv2.waitKey(0)


num_labels, labels_im = cv2.connectedComponents(thresh, connectivity=8)
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
    cv2.waitKey(0)

print("# of bricks (accurate):", len(np.unique(labels_im)))

'''lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

lines = cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength=10,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)'''

#cv2.imshow('image', image)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
