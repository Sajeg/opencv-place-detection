# This Code is based on this: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
import numpy as np
import cv2 as cv

# get Corners of the Image
def get_corners(img_path):
    img = cv.imread(img_path)
    grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    operated_image = np.float32(grey_image)

    return cv.cornerHarris(operated_image, 2, 5, 0.07)

# Save these Corners as arrays
data0 = get_corners("./input/image.jpg")
data1 = get_corners("./input/image2.jpg")
data2 = get_corners("./input/image3.jpg")
data3 = get_corners("./input/image4.jpg")

# then compare two of them
common_elements = np.intersect1d(data1, data2)

print(common_elements)

# and check if the third is also has these corners
if np.all(np.isin(common_elements, data3)):
    print("Yes")

# and check if the fourth picture does not have these corners
if  np.all(np.isin(common_elements, data0)):
    print("No")

# So the Output I hope for is Yes No and what do I get *drum roll* *presses shaking the play button*
# So I got a few errors
# and after fixing them I got: Nothing. I guess I'll need to use another approach