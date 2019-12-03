import cv2
import numpy as np

class Transformer:

    offset = 100 # offset for dst points
    # For source points I'm grabbing the outer four detected corners

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    refPt = []
    points = 0
    resizeScale = 0.5
    scaleFactor = 0.2

    transferMxt = [[ 593,  450], [ 691,  451], [1075,  690], [248,  690]]

    # left_upper_corner = (639, 427)#(568, 470)#(298-2, 230)#(230, 298)
    # right_upper_corner = (653, 427)#(710, 470)#(363-10, 230)#(230, 363)
    # left_bottom_corner = (269, 676)#(210, 710)#(144+20, 340)#(350, 144)
    # right_bottom_corner = (1046, 674)#(1074, 710)#(558-20, 340)#(350, 558)

    left_upper_corner = transferMxt[0]
    right_upper_corner = transferMxt[1]
    left_bottom_corner = transferMxt[3]
    right_bottom_corner = transferMxt[2]

    srcPoints = np.float32([left_upper_corner, right_upper_corner, right_bottom_corner,left_bottom_corner])*resizeScale
    srcPoints = srcPoints*(1/resizeScale)

    img_size = (1280, 720)
    left_upper_corner = (img_size[0]*scaleFactor, offset) #(tsf.offset, img_size[0]*0.2)
    right_upper_corner = (img_size[0]*(1-scaleFactor), offset) #(tsf.offset, img_size[0]*0.8)
    left_bottom_corner = (img_size[0]*scaleFactor, img_size[1]-offset) # (img_size[1]-tsf.offset, img_size[0]*0.2)
    right_bottom_corner = (img_size[0]*(1-scaleFactor), img_size[1]-offset)#(img_size[1]-tsf.offset, img_size[0]*0.8)
    dstPoints = np.float32([left_upper_corner, right_upper_corner, right_bottom_corner, left_bottom_corner])


    def start_gather(self, img):

        width = int(img.shape[1]*self.resizeScale)
        heighth = int(img.shape[0]*self.resizeScale)

        img = cv2.resize(img, (width, heighth))
        self.image = img
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_transform)
        cv2.putText(self.image, "Click Four points to start transform: tl-tr-br-bl", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('image',img)
        cv2.waitKey(0)

    def click_and_transform(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and self.points < 6:
            self.refPt.append((x, y))
            self.points = self.points+1

        if len(self.refPt) > 1 and self.points < 6:
            cv2.line(self.image, self.refPt[-2], self.refPt[-1], (255,0,0), 2)
            cv2.imshow("image", self.image)

        if self.points > 4:
            self.srcPoints = np.float32(self.refPt[0:4])*2
            cv2.putText(self.image, "Press ESC to exit and transform", (200,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            print(self.srcPoints)

    def resetDstPoints(self, img):

        image_size = (img.shape[1], img.shape[0]) # (width, heighth)
        self.img_size = image_size

        left_upper_corner = (image_size[0]*self.scaleFactor, self.offset) #(tsf.offset, img_size[0]*0.2)
        right_upper_corner = (image_size[0]*(1-self.scaleFactor), self.offset) #(tsf.offset, img_size[0]*0.8)
        left_bottom_corner = (image_size[0]*self.scaleFactor, image_size[1]) # (img_size[1]-tsf.offset, img_size[0]*0.2)
        right_bottom_corner = (image_size[0]*(1-self.scaleFactor), image_size[1])#(img_size[1]-tsf.offset, img_size[0]*0.8)
        self.dstPoints = np.float32([left_upper_corner, right_upper_corner, right_bottom_corner, left_bottom_corner])

    def warpImage(self, undistort):

        M = cv2.getPerspectiveTransform(self.srcPoints, self.dstPoints)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undistort, M, self.img_size)

        print("src points:", self.srcPoints)
        print("dst points:", self.dstPoints)

        return warped, M

    def unwarpImage(self, warped):

        Minv = cv2.getPerspectiveTransform(self.dstPoints, self.srcPoints)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(warped, Minv, self.img_size)

        return warped, Minv

