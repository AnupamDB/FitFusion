import cv2
import mediapipe as mp
import numpy as np
import os
import uuid

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a=np.array(a) # First
    b=np.array(b) # Mid
    c=np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle


cap = cv2.VideoCapture(0)

frame_width = 1000
frame_height = 562

correction_factor_width = 0 #- 650
correction_factor_height = 0 #- 300

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


#set the counter
curr_pos = 0
counter = 0
step_counter = 0
counter_lock = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        #Frame resize
        #frame = cv2.resize(frame,(frame_width, frame_height))    
        
        if ret == True:
            
            #BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Flip on horizontal
            #image = cv2.flip(image, 1)

            #Set writeable Flags = False
            image.flags.writeable = False

            #Detections
            results = pose.process(image)

            #Set writeable flags = True
            image.flags.writeable = True

            #RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            lndmrk_hide = [1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32]
            for id, lndmrk in enumerate(landmarks):
                if id in lndmrk_hide:
                    lndmrk.visibility = 0
                    
            #Get Coordinate
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            lshldr = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            rshldr = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            #calculate offset angle
            offset_angle = calculate_angle(lshldr,nose,rshldr)
            
            #Visualise
            cv2.putText(image, str(round(offset_angle,2)), 
                        tuple(np.multiply(nose, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            offset_thresh = None
            if offset_angle<15:
                offset_thresh = True
            else:
                offset_thresh = False
            
            #get coordinates of joints
            lshldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rshldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            lknee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rknee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            lankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            rankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            lelbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            relbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            lwrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            rwrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Feedback back stright alignment
            if lshldr.visibility and lhip.visibility:
                #caculate alignment from left side 
                shldr_coords = [lshldr.x,lshldr.y]
                hip_coords = [lhip.x,lhip.y]
                knee_coords = [lknee.x,lknee.y]
                shldr_vert_angle = calculate_angle([lhip.x,lhip.y],[lshldr.x,lshldr.y],[lshldr.x,0])
                hip_angle = calculate_angle([lshldr.x,lshldr.y],[lhip.x,lhip.y],[lknee.x,lknee.y])
                knee_angle = calculate_angle([lhip.x,lhip.y],[lknee.x,lknee.y],[lankle.x,lankle.y])
            elif rshldr.visibility and rhip.visibility:
                #caculate alignment from left side
                shldr_coords = [rshldr.x,rshldr.y]
                hip_coords = [rhip.x,rhip.y]
                knee_coords = [rknee.x,rknee.y]
                shldr_vert_angle = calculate_angle([rhip.x,rhip.y],[rshldr.x,rshldr.y],[rshldr.x,0])
                hip_angle = calculate_angle([rshldr.x,rshldr.y],[rhip.x,rhip.y],[rknee.x,rknee.y])
                knee_angle = calculate_angle([rhip.x,rhip.y],[rknee.x,rknee.y],[rankle.x,rankle.y])
            else:
                shldr_vert_angle = None
                hip_angle = None
                knee_angle = None
            
            #Visualise
            cv2.putText(image, str(round(shldr_vert_angle,2)), 
                        tuple(np.multiply(shldr_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            cv2.putText(image, str(round(hip_angle,2)), 
                        tuple(np.multiply(hip_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            cv2.putText(image, str(round(knee_angle,2)), 
                        tuple(np.multiply(knee_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

            
            # feedback_thrush = None
            # if shldr_vert_angle>20:
            #     feedback_thresh = True
            # else:
            #     feedback_thresh = False
            
            
            
            # Feedback hand vertical inclination
            shldr_coords = None
            shldr_vert_angle = None
            if lshldr.visibility and lelbow.visibility:
                #caculate left hand-vertical inclination
                shldr_coords = [lshldr.x,lshldr.y]
                shldr_vert_angle = calculate_angle([lelbow.x,lelbow.y],shldr_coords,[lshldr.x,0])
            elif rshldr.visibility and relbow.visibility:
                #caculate right hand-vertical inclination
                shldr_coords = [rshldr.x,rshldr.y]
                shldr_vert_angle = calculate_angle([relbow.x,relbow.y],shldr_coords,[rshldr.x,0])
            else:
                shldr_vert_angle = None
            
            #Visualise
            cv2.putText(image, str(round(shldr_vert_angle,2)), 
                        tuple(np.multiply(shldr_coords, [frame_width + correction_factor_width, frame_height - 100 + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            # feedback_thrush = None
            # if shldr_vert_angle>20:
            #     feedback_thresh = True
            # else:
            #     feedback_thresh = False
                
                
            # Feedback hand motion
            elbow_coords = None
            elbow_angle = None
            if lshldr.visibility and lelbow.visibility:
                #caculate left elbow motion
                elbow_coords = [lelbow.x,lelbow.y]
                elbow_angle = calculate_angle([lshldr.x,lshldr.y],elbow_coords,[lwrist.x,lwrist.y])
            elif rshldr.visibility and relbow.visibility:
                #caculate right elbow motion
                elbow_coords = [relbow.x,relbow.y]
                elbow_angle = calculate_angle([rshldr.x,rshldr.y],elbow_coords,[rwrist.x,rwrist.y])
            else:
                elbow_angle = None
            
            #Visualise
            cv2.putText(image, str(round(elbow_angle,2)), 
                        tuple(np.multiply(elbow_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            # feedback_thrush = None
            # if elbow_angle>20:
            #     feedback_thresh = True
            # else:
            #     feedback_thresh = False
            
            stage_angle = elbow_angle
            if stage_angle > 130 and stage_angle < 170:
                curr_pos = 1
            if stage_angle > 90 and stage_angle < 130:
                curr_pos = 2
            elif stage_angle > 50 and stage_angle < 90:
                curr_pos = 3
                counter_lock = True
            
            if step_counter == (curr_pos-1):
                step_counter = curr_pos
            elif step_counter == (curr_pos+1):
                step_counter = curr_pos
            if counter_lock and (step_counter == 1 and elbow_angle > 160):
                counter += 1
                counter_lock = False
                step_counter = 0
            
            #setup status box
            cv2.rectangle(image, (0,0), (200,100), (225,117,16), -1)
        
            #offset data
            cv2.putText(image, 'REPS: ', (10,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        
            cv2.putText(image, str(counter), (105,55), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            #bar progress
            if stage_angle < 75:
                prgs_val = 0
            else:
                prgs_val = int(stage_angle)
            
            #Interpolating the value
            value = int(np.interp(prgs_val, [55, 155], [210, 400]))
            percent = int(100 * (value - 210)/190)
            percent = 100 - percent
            #setup Progress bor
            cv2.putText(image, str(percent)+'%', (25,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (10,209), (50,400), (0,0,255), 3)
            cv2.rectangle(image, (10, value), (50,400), (0,255,0), -1)
        
            
        except:
            pass
        
        #Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,0), thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(155,155,155), thickness=2,circle_radius=2))
        
        
        
        #video Feed 
        cv2.imshow('Push up', image)
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()