import argparse
import numpy as np
import cv2 as cv
from pathlib import Path
from itertools import chain

# Adapted from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

parser = argparse.ArgumentParser()
parser.add_argument('--in-dir')

parser.add_argument('--rows', type=int, default=9)
parser.add_argument('--cols', type=int, default=6)

args = parser.parse_args()

class Calibrator:
    def __setup_chessboard_points(self):
        points = np.zeros((args.rows * args.cols, 3), np.float32)
        points[:,:2] = np.mgrid[:args.rows,0:args.cols].T.reshape(-1, 2)
        self.chessboard_points = points

    def __init__(self):
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.__setup_chessboard_points()

    def find_corners(self, img):
        self.image_shape = img.shape[::-1]

        found, corners = cv.findChessboardCorners(img, (args.rows, args.cols), None)
        if found:
            corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), self.criteria)
            return (True, corners2)
        else:
            return (False, None)

    def save_corners(self, corners):
        self.objpoints.append(self.chessboard_points)
        self.imgpoints.append(corners)

    def process_image(self, img):
        found, corners = self.find_corners(img)
        if found:
            self.save_corners(corners)

    @property
    def data_count(self):
        return len(self.objpoints)

    def calibrate(self):
        return cv.calibrateCamera(self.objpoints, self.imgpoints, self.image_shape, None, None)

calibrator = Calibrator()

if args.in_dir:
    in_dir = Path(args.in_dir)
    assert in_dir.is_dir()
    for img in chain(in_dir.glob("*.png"), in_dir.glob("*.jpg")):
        mat = cv.imread(str(img))
        gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
        calibrator.process_image(gray)
else:
    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error opening camera for capture")
            exit(1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        found, corners = calibrator.find_corners(gray)
        if found:
            cv.drawChessboardCorners(frame, (args.rows, args.cols), corners, True)

        cv.putText(frame, str(calibrator.data_count), (5, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv.imshow('frame', frame)

        keys = cv.pollKey()
        if keys & 0xFF == ord('q'):
            break
        if keys & 0xFF == ord('c'):
            if found:
                calibrator.save_corners(corners)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = calibrator.calibrate()

if ret:
    print(f'fx: {mtx[0,0]}')
    print(f'fy: {mtx[1,1]}')
    print(f'cx: {mtx[0,2]}')
    print(f'cy: {mtx[1,2]}')
else:
    print("Unable to calibrate camera with provided data")
