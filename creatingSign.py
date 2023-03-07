import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical

class newSign:
    global actions
    actions = []

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_Data')

    # Thirty videos worth of data
    no_sequences = 15

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Folder start
    start_folder = 15

    def __init__(self,sign):
        self.sign = sign
        actions.append(self.sign)

    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_styled_landmarks(image, results):
    # Draw face connections
        newSign.mp_drawing.draw_landmarks(image, results.face_landmarks, newSign.mp_holistic.FACEMESH_TESSELATION, 
                                newSign.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                newSign.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        newSign.mp_drawing.draw_landmarks(image, results.pose_landmarks, newSign.mp_holistic.POSE_CONNECTIONS,
                                newSign.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                newSign.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        newSign.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, newSign.mp_holistic.HAND_CONNECTIONS, 
                                newSign.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                newSign.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        newSign.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, newSign.mp_holistic.HAND_CONNECTIONS, 
                                newSign.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                newSign.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
        
    
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    
    def capture_sign(self):
        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with newSign.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # NEW LOOP
            # Loop through sequences aka videos
            for sequence in range(newSign.no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(newSign.sequence_length):
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = newSign.mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    newSign.draw_styled_landmarks(image, results)
                        
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[-1], sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[-1], sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        
                    # NEW Export keypoints
                    keypoints = newSign.extract_keypoints(results)
                    npy_path = os.path.join(newSign.DATA_PATH, actions, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                            
        cap.release()
        cv2.destroyAllWindows()

    

