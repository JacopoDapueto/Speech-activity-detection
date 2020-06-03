from SpeechDetection.Frame import Frame
from Utils.Utils import createSample
from SpeechDetection.NeuralNetwork.Network import Network
from Utils.Classes import Classes
import numpy as np
from Data.Standardization import Standardization

class SpeechDetector():

    def __init__(self, data, fs, frame_duration, overlap_rate, gender, network, standardization):
        self.data = data
        self.frame_duration = frame_duration
        self.gender = gender
        self.fs = fs # sampling frequency: Number of samples per second
        self.network = network # network already trained and loaded
        self.standardization = standardization # the mean and std are already loaded
        self.Frames = Frame(self.data, fs, frame_duration, overlap_rate)

    def isSpeech(self):
        # assign to each frame if it is speech or non-speech
        zcr_frames = self.Frames.ZCR()
        energy_frames = self.Frames.Energy()
        htn_frames = self.Frames.HTN()
        # sfm_frames = self.frames.SFM()
        magnitude_frames = self.Frames.Magnitude()
        mfcc_frames = self.Frames.MFCC()

        data_to_predict = createSample(zcr_frames, energy_frames, htn_frames, magnitude_frames, mfcc_frames, self.gender)
        data_to_predict = self.standardization.standardizeData(data_to_predict)
        self.labeled_frames = self.network.predictLabel(data_to_predict)

        #self.labeled_frames = np.array(labeled_frames) # list containing the results
        return self.labeled_frames, self.Frames.frame_list
