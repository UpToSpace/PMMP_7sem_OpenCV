import cv2
import numpy as np
import os

# Загрузка уже обученных каскадных классификаторов
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# def detect_medical_mask(image_path):
#
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 1.2, 3)
#
#     for (x, y, w, h) in faces:
#
#         roi_gray = gray[y:y + h, x:x + w]
#
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
#
#         mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 7)
#         for (mx, my, mw, mh) in mouth:
#             cv2.rectangle(img, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 0, 255), 2)
#
#     if len(mouth) > 0:
#         print("Медицинская маска не надета.")
#     else:
#         print("Медицинская маска надета.")
#
#     cv2.imshow('Detected Features', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # 1, 2
#
# detect_medical_mask('1.jpg')
# detect_medical_mask('2.jpg')

# 3
dir = './faces/train'
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
people = []
for i in os.listdir(dir):
    people.append(i)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

dir = './faces/test'
for img_name in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face detection
    face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face_rect:
        face_roi = gray[y: y + h, x: x + w]
        label, confidence = face_recognizer.predict(face_roi)
        print(f'image= {img_name} is {people[label]} with a confidence of {confidence}')
        cv2.putText(img, str(people[label]), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), thickness=2)
    cv2.imshow(img_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()