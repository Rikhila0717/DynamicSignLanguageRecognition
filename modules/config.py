import os
import mediapipe as mp

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Thirty videos worth of data
no_sequences = 15

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 15

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities