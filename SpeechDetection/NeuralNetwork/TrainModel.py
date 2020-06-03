from Data.LoadData import LoadData
from Data.Standardization import Standardization
from SpeechDetection.NeuralNetwork.Network import Network

import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = LoadData().getTrainAndTest()

sd = Standardization()
sd.calculateMeanStd(X_train)
X_train = sd.standardizeData(X_train)
X_test = sd.standardizeData(X_test)
#sd.saveMeanStd()
#sd.loadMeanStd()

print("X_train:{} X_test:{} y_train:{} y_test:{} ".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
net = Network(X_train.shape[1])

# create, compile and fit the model
net.createModel()
net.compileModel()
net.trainModel(X_train, y_train)

# test performances
print("Test balanced accuracy:{} ".format(net.testModel(X_test, y_test)))

# plot history
net.plotHistory()

# saving the model
#net.saveModel()
