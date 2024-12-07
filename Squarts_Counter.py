import cv2
import mediapipe as mp
import numpy as np


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


cap = cv2.VideoCapture('Input_data/squat_input_data.mp4')

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
            
            # Feedback hip-vertical inclination
            lshldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rshldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            hip_coords = None
            hip_vert_angle = None
            if lshldr.visibility and lhip.visibility:
                #caculate left hip-vertical inclination
                hip_coords = [lhip.x,lhip.y]
                hip_vert_angle = calculate_angle([lshldr.x,lshldr.y],hip_coords,[lhip.x,0])
            elif rshldr.visibility and rhip.visibility:
                #caculate left hip-vertical inclination
                hip_coords = [rhip.x,rhip.y]
                hip_vert_angle = calculate_angle([rshldr.x,rshldr.y],hip_coords,[rhip.x,0])
            else:
                hip_vert_angle = None
            
            #Visualise
            cv2.putText(image, str(round(hip_vert_angle,2)), 
                        tuple(np.multiply(hip_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            # feedback_thrush = None
            # if hip_vert_angle>20:
            #     feedback_thresh = True
            # else:
            #     feedback_thresh = False
            
            
            # Feedback knee-vertical inclination
            lknee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rknee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            knee_coords = None
            knee_vert_angle = None
            if lknee.visibility and lhip.visibility:
                #caculate left knee-vertical inclination
                knee_coords = [lknee.x,lknee.y]
                knee_vert_angle = calculate_angle([lhip.x,lhip.y],knee_coords,[lknee.x,0])
            elif rknee.visibility and rhip.visibility:
                #caculate right knee-vertical inclination
                knee_coords = [rknee.x,rknee.y]
                knee_vert_angle = calculate_angle([rhip.x,rhip.y],knee_coords,[rknee.x,0])
            else:
                knee_vert_angle = None
            
            #Visualise
            cv2.putText(image, str(round(knee_vert_angle,2)), 
                        tuple(np.multiply(knee_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            # feedback_thrush = None
            # if knee_vert_angle>20:
            #     feedback_thresh = True
            # else:
            #     feedback_thresh = False
            
            
            
            # Feedback ankle-vertical inclination
            lknee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rknee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            lankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            rankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            ankle_coords = None
            ankle_vert_angle = None
            if lknee.visibility and lankle.visibility:
                #caculate left ankle-vertical inclination
                ankle_coords = [lankle.x,lankle.y]
                ankle_vert_angle = calculate_angle([lknee.x,lknee.y],ankle_coords,[lankle.x,0])
            elif rknee.visibility and rankle.visibility:
                #caculate right ankle-vertical inclination
                ankle_coords = [rankle.x,rankle.y]
                ankle_vert_angle = calculate_angle([rknee.x,rknee.y],ankle_coords,[rankle.x,0])
            else:
                ankle_vert_angle = None
            
            #Visualise
            cv2.putText(image, str(round(ankle_vert_angle,2)), 
                        tuple(np.multiply(ankle_coords, [frame_width + correction_factor_width, frame_height + correction_factor_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        
            feedback_thrush = None
            if ankle_vert_angle>20:
                feedback_thresh = True
            else:
                feedback_thresh = False
            
            stage_angle = knee_vert_angle
            if stage_angle > 10 and stage_angle < 30:
                curr_pos = 1
            if stage_angle > 30 and stage_angle < 60:
                curr_pos = 2
            elif stage_angle > 60 and stage_angle < 88:
                curr_pos = 3
                counter_lock = True
            
            if step_counter == (curr_pos-1):
                step_counter = curr_pos
            elif step_counter == (curr_pos+1):
                step_counter = curr_pos
            if counter_lock and (step_counter == 1 and knee_vert_angle < 8):
                counter += 1
                counter_lock = False
                step_counter = 0
            
            #setup status box
            cv2.rectangle(image, (0,0), (200,100), (225,117,16), -1)
        
            #offset data
            cv2.putText(image, 'REPS: ', (10,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        
            cv2.putText(image, str(counter), (105,55), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            #bar progress
            if stage_angle < 4:
                prgs_val = 0
            else:
                prgs_val = stage_angle
            value = int(np.interp(prgs_val, [0, 100], [210, 400]))
            percent = int(100 * (value - 210)/190)
            #setup Progress bor
            cv2.putText(image, str(percent)+'%', (25,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (30,209), (70,400), (0,0,255), 3)
            cv2.rectangle(image, (30, value), (70,400), (0,255,0), -1)
        
            
        except:
            pass
        
        #Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,0), thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(155,155,155), thickness=2,circle_radius=2))
        
        
        #Save our image
        #cv2.imwrite(os.path.join('Output_data', '{}.jpg'.format(uuid.uuid1())), image)
        
        #video Feed 
        cv2.imshow('Squat', image)
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()