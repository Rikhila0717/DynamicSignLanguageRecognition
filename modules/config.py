import os
import mediapipe as mp

# Path for exported data, numpy arrays
ASL_DATA_PATH = os.path.join('ASL_Data')
ISL_DATA_PATH = os.path.join('ISL_Data')
BSL_DATA_PATH = os.path.join('BSL_Data')
FSL_DATA_PATH = os.path.join('FSL_Data')



# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 15

# Folder start
start_folder = 30

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


