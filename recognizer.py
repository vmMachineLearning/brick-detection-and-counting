import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import measure


def countBricks(imageURL, kernelSize):
    image = cv2.imread(imageURL)
    print("Height: {},Width: {}".format(*image.shape[:2]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blurred = cv2.bilateralFilter(gray, 21, 41, 41)
    edged = cv2.Canny(blurred, 60, 220,
                      apertureSize=3)
   
    thresh = threshold_local(blurred, 21, offset=15).astype("uint8")*255
    mask = cv2.bitwise_not(edged)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernelSize, kernelSize))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_copy = thresh.copy()
    (cnts, _) = cv2.findContours(thresh_copy,
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    for c in cnts:
        cv2.drawContours(output, c, -1, (0, 255, 0), 2)
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
        #cv2.waitKey(0)
    print("# of bricks (accurate):", len(np.unique(labels_im)) - 1)
    # cv2.waitKey(0)
