import cv2
import matplotlib.pyplot as plt
import numpy as np

#1
image = cv2.imread('box.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# параметры детектора углов Харриса
block_size = 2
ksize = 3
k = 0.04

corners = cv2.cornerHarris(gray, block_size, ksize, k)

harris_image = image.copy()
harris_image[corners > 0.01 * corners.max()] = [0, 255, 0]

cv2.imshow('Harris Corners', harris_image)

#2
# параметры детектора углов Ши Томаси
max_corners = 100
quality_level = 0.01
min_distance = 10

corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

corners = np.int0(corners)
shi_image = image.copy()

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(shi_image, (x, y), 3, 255, -1)

cv2.imshow('Shi-Tomasi Corners', shi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3
image = cv2.imread('studio.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
max_corners = 100
quality_level = 0.1
min_distance = 200
corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
corners = np.int0(corners)

pt_A = corners[3][0]
pt_B = corners[4][0]
pt_C = corners[1][0]
pt_D = corners[0][0]
shi_image = image.copy()

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(shi_image, (x, y), 3, 255, -1)
cv2.imshow('Shi-Tomasi Corners', shi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))


input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight],
                        [maxWidth, maxHeight],
                        [maxWidth, 0]])
M = cv2.getPerspectiveTransform(input_pts, output_pts)

out = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
winname = 'Transformed Image'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 40, 30)
cv2.imshow(winname, out)
cv2.waitKey(0)
cv2.destroyAllWindows()