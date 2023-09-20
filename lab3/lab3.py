import cv2

# Открываем видеопоток с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    sobel_x = cv2.Sobel(blurred_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_frame, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

    laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F)

    canny = cv2.Canny(blurred_frame, 50, 150)

    cv2.imshow('Original', frame)
    cv2.imshow('Sobel', sobel_combined)
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('Canny', canny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
