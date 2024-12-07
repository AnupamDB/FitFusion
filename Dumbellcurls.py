import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(img, p1, p2, p3, lmList):
    # Get the landmarks
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Calculate the Angle
    angle = math.degrees(abs(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2)))
    if angle > 180:
        angle = 360-angle

    # Draw the lines and circles
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
    cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase the width of the window
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust the height proportionally

# Curl counter variables
counter = 0 
count = 0
dir = 0
stage = None

# Setup Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lmList = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Calculate angle
                angle = calculate_angle(img, 11, 13, 15, lmList)
                angle_shoulder = calculate_angle(img, 13, 11, 23, lmList)

                # check = np.interp(angle1, (10, 30), (0, 10))
                per = np.interp(angle, (50, 145), (100, 0))
                bar = np.interp(angle, (50, 145), (100, 720))

                # Check for dumbbell curls
                color = (255, 0, 255)
                if angle_shoulder>=0 and angle_shoulder<=20:
                    if per == 100:
                        color = (0, 255, 0)
                        if dir == 0:
                            count += 0.5
                            dir = 1
                    if per == 0:
                        color = (0, 255, 0)
                        if dir == 1:
                            count += 0.5
                            dir = 0

                # Draw the bar
                cv2.rectangle(img, (1200, 100), (1250, 720), color, 3)
                cv2.rectangle(img, (1200, int(bar)), (1250, 720), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                # Draw the curl count
                cv2.rectangle(img, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

        # Convert back to BGR for displaying in OpenCV
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mediapipe Feed', img)

        # Detect if the "X" button is clicked
        if cv2.getWindowProperty('Mediapipe Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
