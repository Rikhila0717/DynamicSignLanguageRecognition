import os
import numpy as np

with open("signslist.text") as f:
    actions = f.readlines()
actions = [x.strip() for x in actions]
print(actions)
actions = np.array(actions)
print(actions.shape)
