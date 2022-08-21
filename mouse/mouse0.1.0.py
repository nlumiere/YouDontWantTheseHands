import mediapipe as mp
import cv2
import keyboard
import pyautogui
import mouse
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

screenw, screenh = pyautogui.size()
print(screenw, screenh)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    frame = 0
    lockout = 0
    prev_x = 0
    prev_y = 0
    while cap.isOpened():
        frame += 1

        # TODO: Frame extrapolation for performance
        # if frame % 3 != 0:
        #     # TODO: Add acceleration
        #     continue

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if keyboard.is_pressed('q'):
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_handedness:
            hand = results.multi_handedness[0].classification[0].label
            if not hand == 'Right':
                continue
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                index_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                v_x = min(max(index_landmark.x, .15), .85)
                v_x = (v_x - .15)*10/7 * screenw
                v_y = min(max(index_landmark.y, .05), .75)
                v_y = (v_y - .05)*10/7 * screenh

                # janky stablization
                if abs(v_x - prev_x) < screenw/200 and abs(v_y - prev_y) < screenh/200:
                    v_x = (v_x + (prev_x * 3))/4
                    v_y = (v_y + (prev_y * 3))/4
                mouse.move(v_x, v_y, absolute=True)
                prev_x = v_x
                prev_y = v_y

                thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                if thumb_landmark.x > index_landmark.x - .05:
                    if lockout == 0:
                        print("CLICK")
                        mouse.click('left')
                        lockout = 9
                    else:
                        lockout -= 1
                else:
                    lockout = max(0, lockout - 1)
                        

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
