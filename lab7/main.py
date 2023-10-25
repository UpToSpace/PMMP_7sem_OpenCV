import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=400, detectShadows=True)

while 1:
    ret, frame = cap.read()
    fg_mask = bg_subtractor.apply(frame)
    _, thresholded = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.putText(frame, 'MOTION!!!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# 2
cap = cv2.VideoCapture(0)
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.1, minDistance=5)
mask = np.zeros_like(old_frame) # Пустая маска для рисования потока

while 1:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)

    k = cv2.waitKey(25)
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()


# 3
tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)
ok, frame = video.read()

bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)

while 1:
    ok, frame = video.read()
    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()