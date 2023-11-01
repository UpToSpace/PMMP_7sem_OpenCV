import cv2
import mediapipe as mp
import pyautogui

capture = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

finger_not_shown_count_left = 0
finger_not_shown_count_right = 0
threshold = 0.05

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Проверка левого указательного пальца
    if results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks.landmark
        for i in range(8, 13, 4):
            if hand_landmarks[i].x < hand_landmarks[i - 3].x and hand_landmarks[i].x < hand_landmarks[i - 1].x - threshold:
                finger_not_shown_count_left = 0
                print("Левый")
                pyautogui.keyDown('ctrl')
                pyautogui.press('right')
                pyautogui.keyUp('ctrl')
            else:
                finger_not_shown_count_left += 1

    # Проверка правого указательного пальца
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks.landmark
        for i in range(8, 13, 4):
            if hand_landmarks[i].x > hand_landmarks[i - 3].x and hand_landmarks[i].x > hand_landmarks[i - 1].x + threshold:
                finger_not_shown_count_right = 0
                print("Правый")
                pyautogui.keyDown('ctrl')
                pyautogui.press('left')
                pyautogui.keyUp('ctrl')
            else:
                finger_not_shown_count_right += 1

    # Если пальцы не были показаны влево/вправо в течение 10 кадров, сбросить счетчики
    if finger_not_shown_count_left >= 10:
        finger_not_shown_count_left = 0
    if finger_not_shown_count_right >= 10:
        finger_not_shown_count_right = 0

    cv2.imshow("Hand Landmarks", image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
