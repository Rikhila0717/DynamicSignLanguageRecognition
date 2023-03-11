import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

class Training:

    DATA_PATH = os.path.join('MP_Data')

    sequence_length = 30

    model = Sequential()

    def __init__(self):
        labels, sequences = self.preprocessing()
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)

    def generate_actions(self):
        with open("signslist.text") as f:
            actions = f.readlines()
        actions = [x.strip() for x in actions]
        actions = np.array(actions)
        return actions

    def preprocessing(self):
        actions = self.generate_actions()
        label_map = {label:num for num, label in enumerate(actions)}
        sequences, labels = [], []
        for action in actions:
            for sequence in np.array(os.listdir(os.path.join(Training.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(Training.sequence_length):
                    res = np.load(os.path.join(Training.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        return labels, sequences
    
    
    def lstm_model(self):
        actions = self.generate_actions()
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        Training.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        Training.model.add(LSTM(128, return_sequences=True, activation='relu'))
        Training.model.add(LSTM(64, return_sequences=False, activation='relu'))
        Training.model.add(Dense(64, activation='relu'))
        Training.model.add(Dense(32, activation='relu'))
        Training.model.add(Dense(actions.shape[0], activation='softmax'))
        Training.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        Training.model.fit(self.X_train, self.y_train, epochs=2000, callbacks=[tb_callback])
        Training.model.summary()
        Training.model.save('action.h5')
        

    def predict_accuracy(self):
        # Training.model.load_weights('action.h5')
        yhat = Training.model.predict(self.X_train)
        ytrue = np.argmax(self.y_train, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue,yhat))


obj = Training()
obj.lstm_model()
obj.predict_accuracy()



    

    

    