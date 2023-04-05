from modules.config import ASL_DATA_PATH
from modules.functions import generate_actions
import numpy as np
import os

actions= generate_actions('asl')
# print(actions)
# for action in actions:
    # print('{} - ',action,os.listdir(os.path.join(ASL_DATA_PATH, action)))
    # print('second print')
    # print('sequences',np.array(os.listdir(os.path.join(ASL_DATA_PATH, action))).astype(int))