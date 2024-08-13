# This Code is based on this: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
import cv2 as cv
import matplotlib.pyplot as plt

# input pictures
img1 = cv.imread('./input/book.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./input/table.jpg',cv.IMREAD_GRAYSCALE)
img3 = cv.imread('./input/image.jpg',cv.IMREAD_GRAYSCALE)
sift = cv.SIFT_create()

# sift corner detection again
# it is cool because the scale doesn't matter, and it also recognizes a corner if it is in another scale
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)

bf = cv.BFMatcher()
# First is the query Image and the second the image where it gets trained on
matches = bf.knnMatch(des1,des2,k=2)
matches2 = bf.knnMatch(des1,des3, k=2)

good = []
for m,n in matches:
    # The lower, the better
    if m.distance < 0.5*n.distance:
        good.append([m])

good2 = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good2.append([m])

# draws the calculations onto the image
img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img5 = cv.drawMatchesKnn(img3,kp3,img2,kp2,good2,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# and shows the images with the matplot library
# didn't that pycharm integrated the plotter.
plt.imshow(img4),plt.show()
plt.imshow(img5),plt.show()