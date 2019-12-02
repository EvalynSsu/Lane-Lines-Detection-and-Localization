from Scripts.Calibration import Calibrator
from Scripts.Transform import Transformer
from Scripts.EdgeDetection import EdgeTransFormer
from Scripts import Polynomial
from Scripts.Lines import lines
import cv2
import numpy as np
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


# Camera Calibration
calibration_Path = "camera_cal/*.jpg"
cb = Calibrator(calibration_Path)
ret, mtx, dist, rvecs, tvecs = cb.findObjPointsAndClib()

tsf = Transformer()


frame = 0
lineDetector = lines()
isStable = False
current_fit = []  # leftfit, rightfit for t-1

def draw_img2img(image):

    global frame, lineDetector, isStable, current_fit, lineDetector
    frame += 1

    undistort = cb.undistorting(image,mtx,dist)
    gray = cv2.cvtColor(undistort,cv2.COLOR_BGR2GRAY)
    edgeT = EdgeTransFormer(gray, undistort)
    binaryRst = edgeT.getBinary()
    tsf.resetDstPoints(binaryRst*255)
    warped_bin, M = tsf.warpImage(binaryRst*255)

    # result = warped_bin

    if isStable==False:
        leftFit, rightFit, polyImg, bias, ploty, left_fitx, right_fitx = Polynomial.fit_polynomial_returnImg(warped_bin)
        result = polyImg
        current_fit = (leftFit, rightFit)

    else:
        searchImg, left_fitx, right_fitx, ploty, bias, leftFit, rightFit = Polynomial.search_around_poly(warped_bin, current_fit[0], current_fit[1])
        result = searchImg
        current_fit = (leftFit, rightFit)

    # Calculate the radius of curvature in pixels for both lane lines
    # left_curverad, right_curverad = Polynomial.measure_curvature_pixels(ploty=ploty, left_fit=leftFit, right_fit=rightFit)
    # print(left_curverad, right_curverad)

    left_curverad, right_curverad, bias = Polynomial.measure_curvature_real(ploty, leftFit, rightFit, bias)
    # print(left_curverad, right_curverad, bias)

    radius = (left_curverad+right_curverad)/2
    isStable = lineDetector.add_rst(detected=True, fit=(leftFit, rightFit), radius=radius, bias=bias, linepix=ploty, frame=frame)

    result = drawlines(warped_bin, undistort, left_fitx, right_fitx, ploty, bias, radius)

    return result

def drawlines(warpedBin, undistorImg, left_fitx, right_fitx, ploty, bias, radius):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedBin).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    if isStable:
        # # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), [0, 255, 0])
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), [130, 20, 0])

    # # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, Minv = tsf.unwarpImage(color_warp)

    # # Combine the result with the original image
    result = cv2.addWeighted(undistorImg, 1, newwarp, 0.3, 0)

    if bias>0:
        textBias = "Vehicle is "+str(round(bias, 2))+"m left the center"
    else:
        textBias = "Vehicle is "+str(round(bias*-1, 2))+"m right the center"

    textRaduis = "Raduis of Curvature = "+str(int(radius))+"m"
    print("\n" + textBias)
    print(textRaduis)
    cv2.putText(result, textRaduis, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, textBias, (50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return result

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = draw_img2img(image)
    return result

def process_Video(filePath, outPath):

    white_output = filePath
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(outPath) # .subclip(41,42)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


def demotest(filePath):

    # test the calibration
    img = cv2.imread(filePath)
    # img = cv2.imread("test_images/straight_lines1.jpg")
    undistort = cb.undistorting(img,mtx,dist)
    cv2.imwrite("demo/distort.jpg",undistort)

    # if you want to manually set rectPoints for transformation by clicking
    # tsf.start_gather(img)
    unwarpedImg, M = tsf.warpImage(undistort)
    cv2.imwrite("demo/transform.jpg", unwarpedImg)

    img_size = (img.shape[1], img.shape[0]) # (width, heighth)
    print(img_size)

    gray = cv2.cvtColor(undistort,cv2.COLOR_BGR2GRAY)
    edgeT = EdgeTransFormer(gray, undistort)
    binaryRst = edgeT.getBinary()
    cv2.imwrite("demo/combo_binary.jpg", binaryRst*255)

    tsf.resetDstPoints(binaryRst*255)
    warped_bin, M = tsf.warpImage(binaryRst*255)
    cv2.imwrite("demo/transform_bin.jpg", warped_bin)

    leftFit, rightFit, polyImg, bias, ploty, left_fitx, right_fitx = Polynomial.fit_polynomial_returnImg(warped_bin)
    cv2.imwrite("demo/polyrst.jpg", polyImg)


    searchImg, left_fitx, right_fitx, ploty, bias, left_fix, right_fit = Polynomial.search_around_poly(warped_bin, leftFit, rightFit)
    cv2.imwrite("demo/polysearch.jpg", searchImg)

    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad = Polynomial.measure_curvature_pixels(ploty=ploty, left_fit=leftFit, right_fit=rightFit)
    # print(left_curverad, right_curverad)

    left_curverad, right_curverad, bais = Polynomial.measure_curvature_real(ploty, leftFit, rightFit, 1)
    # print(left_curverad, right_curverad)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_bin).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), [0, 255, 0])

    # # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, Minv = tsf.unwarpImage(color_warp)

    print("size1:", newwarp.shape, "size2:", undistort.shape)

    # # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    cv2.imwrite("demo/reverse.jpg", result)


# uncomment below code to process a image or a video

# demotest("test_images/straight_lines2.jpg")
process_Video('test_videos_output/test1.mp4', "harder_challenge_video.mp4")