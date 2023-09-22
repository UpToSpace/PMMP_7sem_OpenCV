import cv2
import numpy as np

image = cv2.imread('things.jpg')
filterd_image  = cv2.medianBlur(image, 15)
gray_image = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)
ret,thresh_img = cv2.threshold(gray_image, 104, 255, cv2.THRESH_BINARY)

# 1.	Возьмите изображение нескольких предметов на однородном фоне.
# •	Найдите все контуры на изображении используя функцию поиска контуров cv2.findContours();
contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = image.copy()
cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 2)

# •	Найдите только внешние контуры;
external_contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_external_contours = image.copy()
cv2.drawContours(img_external_contours, external_contours, -1, (255, 0, 0), 2)

# •	Нарисуйте прямоугольники, в которые вписаны предметы на изображении;
image_with_contours = image.copy()
for contour in external_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

# •	Выведите количество предметов;
print(f"Количество предметов: {len(external_contours)}")

# •	Выделите контуры с наибольшей длиной и наибольшей площадью.
contour_with_max_length = max(external_contours, key=lambda x: cv2.arcLength(x, True))
contour_with_max_area = max(external_contours, key=lambda x: cv2.contourArea(x))

longest_contour_image = image.copy()
largest_contour_image = image.copy()

cv2.drawContours(longest_contour_image, [contour_with_max_length], -1, (0, 0, 255), 2)
cv2.drawContours(largest_contour_image, [contour_with_max_area], -1, (0, 0, 255), 2)

cv2.imshow('Binary Image', thresh_img)
cv2.imshow('Contours', img_contours)
cv2.imshow('External Contours', img_external_contours)
cv2.imshow('Rectangles', image_with_contours)
cv2.imshow('Longest Contour', longest_contour_image)
cv2.imshow('Largest Contour', largest_contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.	Подсчитайте количество четырехугольных фигур на рисунке
image = cv2.imread('given.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray_image, 200, 255)
external_contours, _ = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_external_contours = image.copy()
cv2.drawContours(img_external_contours, external_contours, -1, (255, 0, 0), 2)

cv2.imshow('Canny Image', canny_img)
cv2.imshow('Contours', img_external_contours)

count_quadrilaterals = 0

for contour in external_contours:
    epsilon = 0.07 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        if cv2.contourArea(contour) < 10000:
            count_quadrilaterals += 1

print(f'Количество четырехугольных фигур: {count_quadrilaterals}')

cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.	На изображении с прямолинейными объектами найдите линии cv2.HoughLines(),
# а на изображении с окружностями найдите окружности методом cv2.HoughCircles().
# Выведите количество найденных окружностей.
# Найдите линии на изображении

image = cv2.imread('circles.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blured_image = cv2.GaussianBlur(gray_image, (17, 15), 0)
_, binary_image = cv2.threshold(blured_image, 170, 255, cv2.THRESH_BINARY)
cv2.imshow('binary Image', binary_image)
circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, 1, 20,
              param1=60,
              param2=55,
              minRadius=0,
              maxRadius=0)
if circles is not None:
    print(f'Количество найденных окружностей: {len(circles[0])}')
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 0, 255), 2)

cv2.imshow('Circles Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread('squares.jpg')
cv2.imshow('Original', image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray_image, 30, 255)
lines = cv2.HoughLines(canny_img, 1, np.pi / 180, threshold=100)
if lines is not None:
    print(f'Количество найденных линий: {len(lines)}')

# Нарисуйте найденные линии на изображении
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Отобразите изображение с нарисованными линиями
cv2.imshow('Lines Image', image)
cv2.imshow('Canny Image', canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
