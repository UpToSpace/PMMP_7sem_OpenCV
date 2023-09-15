import cv2
import numpy as np

#1
image = cv2.imread('v.jpeg')
cv2.imshow('v.jpeg', image)

# фильтр увеличения контрастности
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

filtered_image = cv2.filter2D(image, -1, kernel)

cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2
# blur
blurred_image = cv2.blur(image, (5, 5))

# GaussianBlur
gaussian_blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# medianBlur
median_blurred_image = cv2.medianBlur(image, 5)

cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)
cv2.imshow('Median Blurred Image', median_blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3
# бинарное изображение
vydra_image = cv2.imread("vydra.jpg");
gray_image = cv2.cvtColor(vydra_image, cv2.COLOR_BGR2GRAY)
th, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# ядро для эрозии и дилатации
kernel = np.ones((5, 5), np.uint8)

# эрозию
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# дилатацию
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#4
# адаптивную бинаризацию
adaptive_threshold = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# адаптивной бинаризации
cv2.imshow('Adaptive Threshold', adaptive_threshold)

# эрозию
eroded_image = cv2.erode(adaptive_threshold, kernel, iterations=1)

# дилатацию
dilated_image = cv2.dilate(adaptive_threshold, kernel, iterations=1)

cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()