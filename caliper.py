# caliper.py

import cv2 as cv
import numpy as np

from resize import resize_image

FILENAME = "images/CCA.jpg"
templates = ["templates/left.jpg", "templates/right.jpg"]



def calibrate(image_path):
    """
    Main calibration function. Takes image path and gathers image data to estimate distance (px)
    """
    # try:
    # Read the image
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if image is None: raise FileNotFoundError("Image could not be read")

    # Optional
    image = np.array(resize_image(image, .25))
    upper = isolate_caliper(image)

    threshholding(upper)
    # Find the teeth
    # find_teeth(upper, templates)

    # except Exception as e:
    #     print(f"An error has occured: {e}")

def threshholding(image):
    # apply right transforms
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # get the contours
    contours, heirarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]

    # find the longest contour (probably the one outlining the caliper)
    largest = 0
    index = 0
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if largest < area:
            largest = area
            index = c
            # cv.drawContours(image, [c], -1, (36,255,12), 3)
    
    print(largest, index)
    perimeter = contours[index]
    cv.drawContours(image, contours, index, (0,0,0), 3)









    canvas = np.zeros(image.shape, np.uint8)
    canvas.fill(255)

    cv.drawContours(canvas, contours, index, (0,0,0), 3)

    cv.imshow('Canvas', canvas)
    # cv.imshow('blur.png', blur)
    # cv.imshow('thresh.png', thresh)
    # cv.imshow('image.png', image)
    cv.waitKey()



def find_teeth(image, templates = ["templates/left.jpg", "templates/right.jpg"]):

    for template in templates:
        # apply grayscale to template and image
        grayT = cv.imread(template, cv.IMREAD_ANYCOLOR)
        grayI = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        grayT = cv.cvtColor(grayT, cv.COLOR_BGR2GRAY)

        if grayT is None: raise FileNotFoundError("Templates could not be found")

        # w, h = grayT.shape[:2]
        method = cv.TM_CCOEFF_NORMED
        result = cv.matchTemplate(grayI, grayT, method)

        cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )

        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

        if (method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        # get 4 corners of "bounding box"
        (startX, startY) = maxLoc
        endX = startX + grayT.shape[1]
        endY = startY + grayT.shape[0]

        return (startX, endX, startY, endY)


def isolate_caliper(image):
    """
    Isolates upper 60% of the frame. This is the caliper region
    """

    wI, hI = image.shape[:2]
    hU = int(hI/3)
    upper = image[0:hU*2, 0:wI]

    return upper


if __name__ == "__main__":
    image_path = FILENAME
    calibrate(image_path)
