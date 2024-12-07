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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        # Recolor image back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lmList = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape  # Use the original frame size
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Calculate angles for both arms
                angle_right = calculate_angle(frame, 12, 14, 16, lmList)  # Right arm (shoulder, elbow, wrist)
                angle_left = calculate_angle(frame, 11, 13, 15, lmList)  # Left arm (shoulder, elbow, wrist)
                shldr_angle_left = calculate_angle(frame, 13, 11, 23, lmList)  # Left arm (elbow, shoulder, hip)
                shldr_angle_right = calculate_angle(frame, 14, 12, 24, lmList)  # Right arm (elbow, shoulder, hip)

                # Ensure angles are mirrored correctly for left arm
                # if angle_left > 180:
                #     angle_left = 360 - angle_left

                # Interpolation for both arms
                per_left = np.interp(angle_right, (70, 125), (100, 0))
                bar_left = np.interp(angle_right, (75, 125), (100, 720))

                per_right = np.interp(angle_left, (70, 125), (100, 0))  # Adjust the range for left arm
                bar_right = np.interp(angle_left, (75, 125), (100, 720))  # Adjust the range for left arm

                # Check for shoulder press movement
                color_right = (255, 0, 255)
                color_left = (255, 0, 255)

                if (shldr_angle_left>=85 and shldr_angle_left<=170) and (shldr_angle_right>=85 and shldr_angle_right<=170):
                    # color_right = (0, 255, 0)
                    # color_left = (0, 255, 0)
                    if per_right == 100 and per_left == 100:
                        color_right = (0, 255, 0)
                        color_left = (0, 255, 0)
                        if dir == 0:
                            count += 0.5
                            dir = 1
                    if per_right == 0 and per_left == 0:
                        color_right = (0, 255, 0)
                        color_left = (0, 255, 0)
                        if dir == 1:
                            count += 0.5
                            dir = 0

                # Draw the bars for both arms
                cv2.rectangle(frame, (1200, 100), (1250, 720), color_right, 3)
                cv2.rectangle(frame, (1200, int(bar_right)), (1250, 720), color_right, cv2.FILLED)
                cv2.putText(frame, f'{int(per_right)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_right, 2)

                cv2.rectangle(frame, (50, 100), (100, 720), color_left, 3)  # Updated position for left bar
                cv2.rectangle(frame, (50, int(bar_left)), (100, 720), color_left, cv2.FILLED)
                cv2.putText(frame, f'{int(per_left)} %', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_left, 2)

                # Draw the curl count
                cv2.rectangle(frame, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

        # Convert back to BGR for displaying in OpenCV
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()