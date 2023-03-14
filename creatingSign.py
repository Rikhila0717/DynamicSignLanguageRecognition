import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append("..")
print("PATH:",sys.path)
from modules.config import DATA_PATH, mp_holistic, mp_drawing, no_sequences, sequence_length
from modules import functions


class newSign:

    def __init__(self,sign):
        self.sign = sign
        # sys.path.append("../static")
        fp = open('static/signslist.text','a')
        print('I opened')
        fp.write(self.sign+',')
        fp.close()
    
    def capture_sign(self):

        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, self.sign, str(sequence)))
            except:
                pass

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # NEW LOOP
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = functions.mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    functions.draw_styled_landmarks(image, results)
                        
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(self.sign, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(self.sign, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        
                    # NEW Export keypoints
                    keypoints = functions.extract_keypoints(results)
                    # print(keypoints)
                    # print(newSign.DATA_PATH, actions[-1], str(sequence), str(frame_num))
                    npy_path = os.path.join(DATA_PATH, self.sign, str(sequence), str(frame_num))
                    # print(npy_path)
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                            
        cap.release()
        cv2.destroyAllWindows()

    
# newSign('hello').capture_sign()
# newSign('thanks').capture_sign()
# newSign('please').capture_sign()
newSign('bye').capture_sign()
