# calibration.py
import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

from resize import resize_image

calibrationDir = ""
referencesDir = ""
backupsDir = ""


def calibrate(showPics=True):
    # Read Image
    root = os.getcwd()
    global calibrationDir 
    calibrationDir = os.path.join(root, 'calibration')
    global referencesDir
    referencesDir = os.path.join(calibrationDir, 'references')
    imgPathList = glob.glob(os.path.join(referencesDir, '*.jpg'))

    # Initialize
    nRows = 6
    nCols = 9
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros( (nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList =[]

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows,nCols), None)
        
        if cornersFound == True:
             worldPtsList.append(worldPtsCur)
             cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
             imgPtsList.append(cornersRefined)

             if showPics:
                 cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                 cv.imshow('Chessboard', imgBGR)
                 cv.waitKey(0)
    cv.destroyAllWindows()

    # Calibrate
    repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
    print('Camera Matrix:\n', camMatrix)
    print('Reproj Error (pixels): {:.4f}'.format(repError))

    # Save Calibration Parameters (later )
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath,
             rep=repError,
             camMatrix=camMatrix,
             distCoeff=distCoeff,
             rvecs=rvecs,
             tvecs=tvecs)
    
    return camMatrix,distCoeff

def removeDistortion(img, camMatrix, distCoeff):
    # root = os.getcwd()
    # imgPath = os.path.join(root,'calibration/checkerboard.jpg')
    # img = cv.imread(imgPath)
    # # img = resize_image(img, .5)

    height,width = img.shape[:2]
    camMatrixNew,roi = cv.getOptimalNewCameraMatrix(camMatrix,distCoeff, (width,height), 1, (width,height))
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    # Draw Line to see Distortion Change
    cv.line(img, (1769, 103), (1780, 922), (255,255,255), 2)
    cv.line(imgUndist, (1769,103), (1780,922), (255,255,255), 2)

    return imgUndist

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    cv.imwrite('Undistorted.jpg', imgUndist)
    cv.imshow("Undistorted", imgUndist)
    cv.waitKey(0)
    cv.destroyAllWindows

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix,distCoeff = calibrate(showPics=True)
    removeDistortion(camMatrix,distCoeff)


def main():
    # camMatrix, distCoeff = calibrate()
    # removeDistortion(camMatrix, distCoeff)
    runRemoveDistortion()


if __name__ == "__main__":
    main()