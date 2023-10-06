import os
import numpy as np
import cv2

dir = './faces/train'
people = []
labels = []
features = []

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for i in os.listdir(dir):
    people.append(i)
print(f'People: {people}')

def create_train():
    i = 0
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            i = i + 1
            img_array = cv2.imread(img_path)
            if img_array is not None:
                print(f'file is: {img_path} i = {i}')
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in face_rect:
                    face_roi = gray[y: y + h, x: x + w]
                    features.append(face_roi)
                    labels.append(label)
            else:
                print(f'file is damaged: {img}')

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

print(f'features length = {len(features)}')
print(f'labels length = {len(labels)}')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
