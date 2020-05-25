import librosa as lb
from librosa.feature import zero_crossing_rate
from librosa.core import load
import numpy as np
import scipy as sc
import scipy.io.wavfile as wf

class SpeechDetector():


    def __init__(self, data, fs):
        self.data = data
        self.fs = fs # sampling frequency: Number of samples per second
        self.windowed_frames()

    # SHORT TIME ANALISYS
    def split_into_frames(self, frame_length, overlap_rate):
        # The audio is divided into frames
        shift = 1 - overlap_rate
        frame_shift = round(self.frame_length * shift)  # shift length in samples: hop_length
        frame_list = lb.util.frame(x = self.data, frame_length= self.frame_length, hop_length= frame_shift, axis=0)  # matrix: each row represents a frame
        return frame_list

    def windowed_frames(self, duration= 30, overlap_rate= 0.5, window_type= "hamm"):
        # The audio is divided into frames and a windows is applied to each of them
        """
        :param duration: frame duration in msec
        :param overlap_rate: how much a frame overlap with the next one (percentage)
        :param window_type: the type of of window applied to each frame
        """
        self.frame_length = int(np.floor(duration * self.fs / 1000))  # frame length (in samples) s -> msec

        frame_list = self.split_into_frames(self.frame_length, overlap_rate)
        window_filter = lb.filters.get_window(window=window_type, Nx=self.frame_length, fftbins=False)

        self.windowed_frame_list = np.multiply(frame_list, window_filter)#[np.multiply(frame, window_filter) for frame in frame_list] # windowing the frames


    def normalize_data(self):
        # normalize the data
        self.data = self.data / max(self.data)

    def ZCR(self):
        # return the zero crossing rate for each (overlapping) frame
        frameWind_list = self.windowed_frame_list
        return [ np.sum(abs(np.diff(np.sign(frameWind-np.mean(frameWind)))))/(2*self.frame_length) for frameWind in frameWind_list]

    def Energy(self):
        # return the energy for each (overlapping) frame
        frameWind_list = self.windowed_frame_list
        return [sum(frameWind ** 2)/len(frameWind) for frameWind in frameWind_list]

    def MSE_Energy(self):
        # return the mean square error of the energy for each (overlapping) frame
        energy_list = self.Energy()
        return [np.sqrt(energy) for energy in energy_list]

    def isSpeech(self):
       # assign to each frame if it is speech or non-speech
       threshold_zcr = 0.2 # per ora molto a caso
       threshold_energy = 0.2 # per ora a casissimo

       frameWind_list = self.windowed_frame_list
       zcr_list = self.ZCR()
       energy_list = self.Energy()

       isSpeech_list = []

       # per ora molto base schifo
       for frameWind, zcr, energy in zip(frameWind_list, zcr_list, energy_list):

           if zcr > threshold_zcr and energy > threshold_energy:
               # is speech
               isSpeech_list.append([ 1 for _ in frameWind])
           else:
               # is non-speech
               isSpeech_list.append([ 0 for _ in frameWind])

       self.isSpeech_list = np.array(isSpeech_list).flatten() # list containing the results

fs, data = wf.read('FSA1.wav')
#data, fs = load(lb.util.example_audio_file())
print(fs)
iss = SpeechDetector(data, fs)
