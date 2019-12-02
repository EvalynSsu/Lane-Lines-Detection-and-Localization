import pickle
import cv2
import numpy as np
import glob

class Calibrator:

    objpoints = []
    imgpoints = []
    corners_v = 6
    corners_h = 9
    filePath = ""

    # prepare object points like (0,0,0) (1,0,0) (2,0,0)...(7,5,0)
    objp = np.zeros((corners_v*corners_h,3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_h,0:corners_v].T.reshape(-1,2)

    # Read in the saved objpoints and imgpoints
    # dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
    # objpoints = dist_pickle["objpoints"]
    # imgpoints = dist_pickle["imgpoints"]

    def __init__(self, filePath):
        self.filePath = filePath

    def findObjPointsAndClib(self):

        images = glob.glob(self.filePath)

        for fname in images:

            # read each image in BGR and convert to gray
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the Chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.corners_h,self.corners_v), None)

            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(self.objp)

                # draw chessboard corners for testing
                # tst_img = cv2.drawChessboardCorners(img, (self.corners_h,self.corners_v), corners, ret)
                # cv2.imwrite("chessboarder.jpg",tst_img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def undistorting(self, img, mtx, dist):
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst


    def setCotners(self, v, h):
        self.corners_h = h
        self.corners_v = v
        self.objp = np.zeros((self.corners_v*self.corners_h,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.corners_h,0:self.corners_v].T.reshape(-1,2)
