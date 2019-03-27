# -*- coding: utf-8 -*-
import cv2
import numpy
import os


img_path = 'C:/Users/ratho/Desktop/AI_2/test_images/test.jpg'

#read the image
image = cv2.imread(img_path)

#Resizing the image
image = cv2.resize(image,(500,400))

#Display  the image
cv2.imshow("Original image", image)


#RGB to Gray 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray_filter = cv2.bilateralFilter(gray, 11, 25, 25) #params(image, )

#Histogram equalization
equalized_image = cv2.equalizeHist(gray_filter)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
morphed_image = cv2.morphologyEx(equalized_image, cv2.MORPH_OPEN, kernel)
"""
MORPH_RECT is used to highlight the rectangular edges and in the output those
rectangles are more prominenet.Kernel builds an array depending on the parameters passed,
in this case : (1,2)
"""


subtracted_image = cv2.subtract(morphed_image, equalized_image)
"""
.subtract will subtract some features of morphed image from equalized image
"""


_, threshold_image = cv2.threshold(morphed_image, 200, 255, cv2.THRESH_BINARY)
"""
If pixel value is greater than a threshold value, it is assigned one value (may be white),
else it is assigned another value (may be black). The function used is cv.threshold. 
First argument is the source image, which should be a grayscale image. 
Second argument is the threshold value which is used to classify the pixel values. 
Third argument is the maxVal which represents the value to be given if pixel value is 
more than (sometimes less than) the threshold value.
The type of thresholding done is binary so we specify it by cv2.THRESH_BINARY.
"""


# Find Edges of the filtered grayscale image

edged_image = cv2.Canny(threshold_image, 160, 200, L2gradient=True) #params(image, threshold1, threshold2)
"""
Hysteresis Thresholding
This stage decides which are all edges are really edges and which are not.
For this, we need two threshold values, minVal and maxVal. 
Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. 
Those who lie between these two thresholds are classified edges or non-edges based on their connectivity.
If they are connected to "sure-edge" pixels, they are considered to be part of edges. 
Otherwise, they are also discarded.
"""

dilation_image = cv2.dilate(edged_image, kernel, iterations = 1)
"""
A pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
So it increases the white region in the image or size of foreground object increases.
"""

(new, cnts, _) = cv2.findContours(dilation_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

NumberPlate = []
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  #This is our approx Number Plate Contour
            break
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)

cv2.imshow("1 - Grayscale conversion", image)

cv2.waitKey(10000)
cv2.destroyAllWindows()
