import os
from calibration import removeDistortion, calibrate
import cv2 as cv

import glob

camMatrix, distCoeff = calibrate(False)

root = os.getcwd()
disorted_dir = os.path.join(root, 'calibration/references')
undistortedDir = os.path.join(root, 'undistorted')

imgPathList = glob.glob(os.path.join(disorted_dir, '*.jpg'))

#Undistort our image of interest
testPath = os.path.join(root, 'test.jpg')
testBGR = cv.imread(testPath)
testUndistorted = removeDistortion(testBGR, camMatrix, distCoeff)
cv.imwrite('TEST UNDISTORTED.jpg', testUndistorted)
cv.waitKey(0)
cv.destroyAllWindows

for imagePath in imgPathList:
    imgBGR = cv.imread(imagePath)
    undistorted = removeDistortion(imgBGR, camMatrix, distCoeff)

    # cv.imshow('image', undistorted)
    # cv.imshow("Undistorted", imgUndist)
    cv.waitKey(0)
    cv.destroyAllWindows

    fileName = os.path.basename(imagePath)

    path = f"{undistortedDir}/{fileName}"

    print(path)

    cv.imwrite(path, undistorted)

