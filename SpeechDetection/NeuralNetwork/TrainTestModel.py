from Data.LoadData import LoadData
from Data.Standardization import Standardization
from SpeechDetection.NeuralNetwork.Network import Network
import numpy as np
'''
File used to load the data, save the standardization parameters and save the trained neural network.
The GUI assumes that the file corresponding to network are present.
'''

print("Loading data...")
X_train, X_test, y_train, y_test = LoadData().getTrainAndTest()

sd = Standardization()
sd.calculateMeanStd(X_train)
X_train = sd.standardizeData(X_train)
X_test = sd.standardizeData(X_test)
sd.saveMeanStd()

# Statistic about the data
print("X_train:{} X_test:{} y_train:{} y_test:{} ".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

_, Train_counts = np.unique(y_train, return_counts= True)
_, Test_counts = np.unique(y_test, return_counts= True)
print("Training set:: Speech:{}, Nonspeech:{} ".format(Train_counts[1], Train_counts[0]))
print("Test set:: Speech:{}, Nonspeech:{} ".format(Test_counts[1], Test_counts[0]))
print("Total:: Speech:{}, Nonspeech:{} ".format(Train_counts[1] + Test_counts[1],Train_counts[0] + Test_counts[0]))
net = Network()

# create, compile and fit the model
net.createModel()
net.compileModel()
print("Training...")
net.trainModel(X_train, y_train)

# test performances
print("Test balanced accuracy:{} ".format(net.testModel(X_test, y_test)))

# plot history
net.plotHistory()

# saving the model
net.saveNetwork()
