# This Code is based on this: https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
import numpy as np
import cv2 as cv

def get_corners(img_path):
    sift = cv.SIFT_create()
    img = cv.imread(img_path)
    grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return sift.detect(grey_image, None)

# Save these Corners as arrays (again) But this time I sort them smart
data1 = get_corners("./input/image2.jpg")
data2 = get_corners("./input/image3.jpg")

# For testing
data3 = get_corners("./input/image4.jpg")
data0 = get_corners("./input/image.jpg")

# then compare two of them but this time it is more complicated
# That's why I decided to switch to ./match.py
