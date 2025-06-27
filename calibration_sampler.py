#webcam control.py 
import datetime
import shutil
import numpy as np
import cv2 as cv
import os

indexer = 0
calibrationDir = ""
referencesDir = ""
backupsDir = ""

def gatherCalibrationData(wipe = True):
    # Access the camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Establish paths
    root = os.getcwd()
    global calibrationDir
    calibrationDir = os.path.join(root, 'calibration')
    global referencesDir 
    referencesDir = os.path.join(calibrationDir, 'references')
    global backupsDir
    backupsDir = os.path.join(calibrationDir, 'backups')

    # If directories do not already exist, make them exist
    os.makedirs(calibrationDir, exist_ok=True)
    os.makedirs(referencesDir, exist_ok=True)
    os.makedirs(backupsDir, exist_ok=True)

    # Clear out any left over calibration data
    if wipe: cleanupCalibration(referencesDir)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv.imshow('frame', gray)

        cv.imshow("Press a key", gray)
        key = cv.waitKey(1) & 0xFF  # Mask for compatibility

        if key != 255:  # If a key was pressed (255 means no key)
            if userCMD_callback(key, frame):
                break


        if cv.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def cleanupCalibration(backup = False):
    # cleanout directory
    if backup:
        # delete each file and unlink it from anything
        for filename in os.listdir(referencesDir):
            file_path = os.path.join(referencesDir, filename)
            try:
                # file or symlink
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # directory/its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
        print("Old Calibration Data Cleared...")
    else:
        # Optional: create a timestamped subfolder inside backups/
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backupDir = os.path.join(backupsDir, f"backup_{timestamp}")
        os.makedirs(backupDir)

        # Move everything
        for filename in os.listdir(referencesDir):
            src_path = os.path.join(referencesDir, filename)
            dst_path = os.path.join(backupDir, filename)
            try:
                shutil.copy(src_path, dst_path)
                print(f"Backed up: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Failed to copy {src_path}. Reason: {e}")

        print(f"Backup complete. Files moved to: {backupDir}")


def userCMD_callback(key, frame):
    global indexer
    if key == ord('q'):
        print("Exiting...")
        return True  # Signal to exit
    elif key == ord('s'):
        print("Saving...")
        # capture the image
        letter = chr(ord('a')+indexer)
        fileName = f"checkerboard_{letter}.jpg"
        fullPath = os.path.join(referencesDir, fileName)
        cv.imwrite(fullPath, frame)

        indexer += 1
        print(f"There are {indexer} photos in {referencesDir}.")
    else:
        print(f"Unknown command: {chr(key)}")
    return False  # Don't exit



if __name__ == "__main__":
    gatherCalibrationData()