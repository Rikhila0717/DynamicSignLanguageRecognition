import numpy as np
import os
import joblib
import pickle
import sys
# sys.path.append('..')
from modules.functions import readLabelsFromS3

from modules.functions import generate_actions
from modules.config import ASL_DATA_PATH,ISL_DATA_PATH,BSL_DATA_PATH,FSL_DATA_PATH,sequence_length
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import L1,L2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L2,L1,L1L2
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize

class Training:

    model = Sequential()

    def __init__(self,lang):
        # print("Hi I started")
        self.lang = lang
        self.actions = generate_actions(self.lang)
        if self.lang=='asl':
            self.DATA_PATH = ASL_DATA_PATH
        elif self.lang=='isl':
            self.DATA_PATH = ISL_DATA_PATH
        elif self.lang=='bsl':
            self.DATA_PATH = BSL_DATA_PATH
        elif self.lang=='fsl':
            self.DATA_PATH = FSL_DATA_PATH
        labels, sequences = self.preprocessing()
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        print(X.shape)
        # print(y.shape)
        X=X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))
        print(X.shape)

        imputer = SimpleImputer(strategy='constant',fill_value=0)
        imputer.fit_transform(X,y)
        X = normalize(X)
        X=X.reshape(X.shape[0],15,1662)
        print(X.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)

    def preprocessing(self):
        label_map = {label:num for num, label in enumerate(self.actions)}
        sequences, labels = [], []
        for action in self.actions:
            for sequence in np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    # print("I'm here trying to read {} of {} of {}".format(frame_num,sequence,action))
                    # res = readLabelsFromS3(self.lang+'data-set','{}data-set/{}/{}/{}.pkl'.format(self.lang,action,sequence,frame_num))
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    # print(res)
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        print('retrieved data')
        return labels, sequences
    
    
    def lstm_model(self):
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        ##our model
        Training.model.add(LSTM(64, return_sequences=True, input_shape=(15,1662)))
        # Training.model.add(LSTM(64, return_sequences=True, activation='tanh'))
        Training.model.add(LSTM(64, return_sequences=False, activation='tanh',activity_regularizer=L1L2(0.01)))
        Training.model.add(Dropout(0.2))
        # Training.model.add(Dense(64, activation='sigmoid'))
        Training.model.add(Dense(32, activation='sigmoid'))
        Training.model.add(Dense(self.actions.shape[0], activation='softmax'))

        # fp_model = "savedModel.sav"
        # print(fp_model)
        # pickle.dump(Training.model, ope n(fp_model,'wb'))
        # return fp_model
        

        ##base paper model

        # Training.model.add(LSTM(64, return_sequences=True,activation='tanh',input_shape=(15,1662)))
        # Training.model.add(LSTM(128, return_sequences=True, activation='tanh'))
        # Training.model.add(Dropout(0.2))
        # Training.model.add(LSTM(64, return_sequences=False, activation='tanh'))
        # Training.model.add(Dropout(0.2))
        # Training.model.add(Dense(64, activation='sigmoid',kernel_regularizer = L1L2(0.01)))
        # Training.model.add(Dense(32, activation='sigmoid'))
        # Training.model.add(Dense(self.actions.shape[0], activation='softmax'))

        Training.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        Training.model.fit(self.X_train, self.y_train, epochs=1500, callbacks=[tb_callback])
        Training.model.summary()
        Training.model.save(self.lang+'model.h5')
        

    def predict_accuracy(self):
        model = load_model(self.lang+'model.h5')
        yhat = model.predict(self.X_train)
        ytrue = np.argmax(self.y_train, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue,yhat))

# obj = Training('asl')
# obj.lstm_model()
# obj.predict_accuracy()



# loaded_model = pickle.load(open(fp_model,'rb'))
# result = loaded_model.score(obj.X_test, obj.y_test)
# print(result)

# asl_obj = Training('asl')
# print('ASL MODEL epochs')
# asl_obj.lstm_model()
# print("ASL model accuracy")
# asl_obj.predict_accuracy()

# bsl_obj = Training('bsl')
# print('BSL model epochs')
# bsl_obj.lstm_model()
# print("BSL model accuracy")
# bsl_obj.predict_accuracy()

isl_obj = Training('isl')
print('ISL model epochs')
isl_obj.lstm_model()
print("ISL model accuracy")
isl_obj.predict_accuracy()

# fsl_obj = Training('fsl')
# print('FSL model epochs')
# fsl_obj.lstm_model()
# print("FSL model accuracy")
# fsl_obj.predict_accuracy()




    

    