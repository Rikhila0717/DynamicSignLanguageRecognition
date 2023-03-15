import numpy as np
import os
import joblib
import pickle

from modules.functions import generate_actions
from modules.config import ASL_DATA_PATH,ISL_DATA_PATH,BSL_DATA_PATH,sequence_length
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

class Training:

    model = Sequential()

    def __init__(self,lang):
        self.lang = lang
        self.actions = generate_actions(self.lang)
        if self.lang=='asl':
            self.DATA_PATH = ASL_DATA_PATH
        elif self.lang=='isl':
            self.DATA_PATH = ISL_DATA_PATH
        elif self.lang=='bsl':
            self.DATA_PATH = BSL_DATA_PATH
        labels, sequences = self.preprocessing()
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)

    def preprocessing(self):
        label_map = {label:num for num, label in enumerate(self.actions)}
        sequences, labels = [], []
        for action in self.actions:
            for sequence in np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        return labels, sequences
    
    
    def lstm_model(self):
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        Training.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        Training.model.add(LSTM(128, return_sequences=True, activation='relu'))
        Training.model.add(LSTM(64, return_sequences=False, activation='relu'))
        Training.model.add(Dense(64, activation='relu'))
        Training.model.add(Dense(32, activation='relu'))
        Training.model.add(Dense(self.actions.shape[0], activation='softmax'))
        Training.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        Training.model.fit(self.X_train, self.y_train, epochs=2000, callbacks=[tb_callback])
        Training.model.summary()
        Training.model.save(self.lang+'model.h5')
        # fp_model = "savedModel.sav"
        # print(fp_model)
        # pickle.dump(Training.model, open(fp_model,'wb'))
        # return fp_model

        

    def predict_accuracy(self):
        model = load_model('static/'+self.lang+'model.h5')
        yhat = model.predict(self.X_train)
        ytrue = np.argmax(self.y_train, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue,yhat))

obj = Training('asl')
obj.lstm_model()
obj.predict_accuracy()



# loaded_model = pickle.load(open(fp_model,'rb'))
# result = loaded_model.score(obj.X_test, obj.y_test)
# print(result)

obj = Training('bsl')
obj.lstm_model()
obj.predict_accuracy()


    

    