#importing dependencies

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#access webcam using opencv

#0 captures input from the webcam
cap = cv2.VideoCapture(0)

#loop through all existing frames
while cap.isOpened():

        # Read feed - grabs the current frame
        ret, frame = cap.read()
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


