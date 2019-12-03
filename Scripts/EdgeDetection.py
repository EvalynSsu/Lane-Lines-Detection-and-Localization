import numpy as np
import cv2

class EdgeTransFormer:

    thresh_min = 20
    thresh_max = 100
    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    def __init__(self, gray, image):

        self.sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        self.sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        self.gray = gray
        self.image = image


    def getBinary(self, sobel_kernel=3, mag_thresh_gradx=(20, 100), mag_thresh_grady=(20, 100), mag_thresh_magBin=(40, 100), mag_thresh_dir=(0.7, 1.3)):

        # Apply each of the thresholding functions
        gray = self.gray
        image = self.image
        self.ksize = sobel_kernel

        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=self.ksize, thresh=mag_thresh_gradx)
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=self.ksize, thresh=mag_thresh_grady)
        mag_binary = self.mag_thresh(gray, sobel_kernel=9, mag_thresh=mag_thresh_magBin)
        dir_binary = self.dir_threshold(gray, sobel_kernel=15, thresh=mag_thresh_dir)
        S_binary = self.get_S(image)

        # for trying different parameters
        # cv2.imwrite("edge/combo_1.jpg", gradx*255)
        # cv2.imwrite("edge/combo_2.jpg", grady*255)
        # cv2.imwrite("edge/combo_3.jpg", mag_binary*255)
        # cv2.imwrite("edge/combo_4.jpg", dir_binary*255)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (S_binary == 1)] = 1
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined



    def getBinary_debug(self, sobel_kernel=3, mag_thresh_gradx=(20, 100), mag_thresh_grady=(20, 100), mag_thresh_magBin=(40, 100), mag_thresh_dir=(0.7, 1.3)):

        # Apply each of the thresholding functions
        gray = self.gray
        image = self.image
        self.ksize = sobel_kernel

        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=self.ksize, thresh=mag_thresh_gradx)
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=self.ksize, thresh=mag_thresh_grady)
        mag_binary = self.mag_thresh(gray, sobel_kernel=9, mag_thresh=mag_thresh_magBin)
        dir_binary = self.dir_threshold(gray, sobel_kernel=15, thresh=mag_thresh_dir)
        S_binary = self.get_S(image)

        # for trying different parameters
        # cv2.imwrite("edge/combo_1.jpg", gradx*255)
        # cv2.imwrite("edge/combo_2.jpg", grady*255)
        # cv2.imwrite("edge/combo_3.jpg", mag_binary*255)
        # cv2.imwrite("edge/combo_4.jpg", dir_binary*255)

        combined_0 = np.zeros_like(dir_binary)
        combined_1 = np.zeros_like(dir_binary)
        combined_2 = np.zeros_like(dir_binary)
        combined_3 = np.zeros_like(dir_binary)

        combined_0[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & (S_binary == 1)] = 1
        combined_1[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (S_binary == 1)] = 1
        combined_2[((gradx == 1) & (grady == 1))] = 1
        combined_3[((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined_0, combined_1, combined_2, combined_3

    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        if orient == 'x':
            scaled_sobel = scaled_sobelx
        else:
            scaled_sobel = scaled_sobely
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.thresh_min) & (scaled_sobel <= self.thresh_max)] = 1
        grad_binary = sxbinary
        return grad_binary

    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        # Grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # Return the binary image
        return dir_binary

    def get_S(self, image):

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        S = hls[:,:,2]

        thresh = (90, 255)
        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1
        return binary





