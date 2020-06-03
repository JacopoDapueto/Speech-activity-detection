import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from keras.models import model_from_json
from  tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Network():
    # Implementation of a simple neural network

    def __init__(self, input_shape, optimizer='adam', loss='binary_crossentropy', metrics='accuracy'):
        self.trained = False
        self.model = Sequential()
        self.input_shape = input_shape
        self.model_path = "Model\\model.json"
        self.weights_path = "Weights\\weights.h5"
        self.history_path = "History\\historyDict"
        self.checkpoint_path = "Checkpoint/cp.ckpt"
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics


    def createModel(self):
        # Add a Dense layer with 32 neurons, with relu as activation function and input dimension equal to the number of features
        self.model.add(Dense(32, input_shape=(self.input_shape,), activation='relu'))
        # To produce the output Add a Dense layer with 1 neurons, with sigmoid as activation function
        self.model.add(Dense(1, activation='sigmoid'))

    def compileModel(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def createCheckPoint(self):
        es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)
        return es_callback, cp_callback

    def trainModel(self, X, y, batch_size= 128, epochs= 100):
        es_callback, cp_callback = self.createCheckPoint()
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.3, callbacks=[es_callback, cp_callback])
        self.trained = True

    def testModel(self, Xtest, ytest):
        if not self.trained:
            raise Exception("The network is not trained yet!")
        y_pred = (self.model.predict(Xtest) > 0.5).astype("int32")
        return balanced_accuracy_score(y_pred, ytest)

    def predictLabel(self, data):
        """
        :param data: matrix where each row represents a frame with its own features
        :return: for each frame the label SPEECH/NONSPEECH
        """
        if not self.trained:
            raise Exception("The network is not trained yet!")
        return self.model.predict_classes(data)

    def saveModel(self):
        model_json = self.model.to_json()
        with open(self.model_path, "w") as json_file:
            json_file.write(model_json)

    def loadModel(self):
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

    def saveWeights(self):
        if not self.trained:
            raise Exception("The network is not trained yet!")
        self.model.save_weights(self.weights_path)

    def loadWeights(self):
        self.model.load_weights(self.weights_path)

    def loadNetwork(self):
        self.loadModel()
        self.loadWeights()
        self.compileModel()
        self.trained = True

    def saveHistory(self):
        if not self.trained:
            raise Exception("The network is not trained yet!")
        with open(self.history_path, 'wb') as file_pi:
            pickle.dump(self.history, file_pi)

    def loadHistory(self):
        self.history = pickle.load(open(self.history_path, "rb"))

    def plotHistory(self):
        if self.history == None:
            raise Exception("The network is not trained yet")

        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.grid()
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.grid()
        plt.show()

