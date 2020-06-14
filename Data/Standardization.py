import numpy as np
import pickle
import os
FILELENGTH=18
class Standardization():

    def __init__(self):
        self.loaded = False  # if the mean and std were loaded from file
        self.m_path = os.path.join(str(__file__)[:-FILELENGTH],  "MeanTrainSet")
        self.s_path = os.path.join(str(__file__)[:-FILELENGTH],  "STDTrainSet")

    def calculateMeanStd(self, X):
        self.m = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)
        self.loaded = True

    def standardizeData(self, X):
        if not self.loaded:
            raise Exception("Mean and Std are not available")
        Xnorm = np.divide((X - self.m), self.s)
        return Xnorm

    def saveMeanStd(self):
        # save mean and std in a file
        if not self.loaded:
            raise Exception("Mean and Std are not available")

        with open(self.m_path, 'wb') as f:
            pickle.dump(self.m, f)

        with open(self.s_path, 'wb') as f:
            pickle.dump(self.s, f)

    def loadMeanStd(self):
        # load mean and std in a file
        with open(self.m_path, 'rb') as f:
            self.m = pickle.load(f)

        with open(self.s_path, 'rb') as f:
            self.s = pickle.load(f)
        self.loaded = True