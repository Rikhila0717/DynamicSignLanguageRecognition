import cv2
import numpy as np
import os
from modules.config import ASL_DATA_PATH,ISL_DATA_PATH,BSL_DATA_PATH,FSL_DATA_PATH, mp_holistic, no_sequences, sequence_length
from modules import functions

class newSign:

    def __init__(self,lang,sign):
        self.sign = sign
        self.lang = lang

        if self.lang=='asl':
            self.DATA_PATH = ASL_DATA_PATH
        elif self.lang=='isl':
            self.DATA_PATH = ISL_DATA_PATH
        elif self.lang=='bsl':
            self.DATA_PATH = BSL_DATA_PATH
        elif self.lang=='fsl':
            self.DATA_PATH = FSL_DATA_PATH

        fp = open('static/'+lang+'signs.text','a')
        fp.write(self.sign+',')
        fp.close()
    
    def capture_sign(self):

        #only for creating local directories for signs
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(self.DATA_PATH, self.sign, str(sequence)))
            except:
                pass

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # NEW LOOP
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # object = s3.Object(self.lang+'-data',self.sign)
                # object.put(Body='hello_hi')
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
                    
                    #for creating a local copy of the sign
                    npy_path = os.path.join(self.DATA_PATH, self.sign, str(sequence), str(frame_num))
                    # print(npy_path)

                    #for saving the sign in the local system
                    np.save(npy_path, keypoints)

                    #to save the sign in Amazon s3
                    # functions.saveLabelsToS3(keypoints,self.lang+'-data','{}-data/{}/{}/{}.pkl'.format(self.lang,self.sign,sequence,frame_num))

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                            
        cap.release()
        cv2.destroyAllWindows()


#to create a sign, pass the language name and the sign name and call the capture_sign() method

#newSign(lang_name,sign_name).capture_sign()