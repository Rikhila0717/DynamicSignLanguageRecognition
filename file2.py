from creatingSign import newSign, actions
from training import Training
import cv2
import numpy as np


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.25

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with newSign.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = newSign.mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        newSign.draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = newSign.extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]
        
        if len(sequence) == 30:
            res = Training.model.predict(np.expand_dims(sequence, axis=0))[0]
            '''print(actions[np.argmax(res)])'''
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
            
            # Viz probabilities
            '''image = file.prob_viz(res, file1.actions, image, file1.colors)'''
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

