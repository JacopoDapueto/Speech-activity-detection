import librosa as lb
from librosa import feature
from Signal_Analysis.features import signal
import numpy as np

class Frame():

    def __init__(self, data, fs, frame_duration, overlap_rate):
        self.data = data
        self.fs = fs
        self.windowed_frames(duration=frame_duration, overlap_rate= overlap_rate, window_type="hamm")

    # SHORT TIME ANALISYS
    def split_into_frames(self, frame_length, overlap_rate):
        # The audio is divided into frames
        shift = 1.0 - overlap_rate
        frame_shift = round(frame_length * shift)  # shift length in samples: hop_length
        frame_list = lb.util.frame(x=self.data, frame_length=self.frame_length, hop_length=frame_shift, axis=0)  # matrix: each row represents a frame
        return frame_list

    def windowed_frames(self, duration=32, overlap_rate=0.0, window_type="hamm"):
        """
        The audio is divided into frames and a windows is applied to each of them

        :param duration: frame duration in msec
        :param overlap_rate: how much a frame overlap with the next one (percentage)
        :param window_type: the type of of window applied to each frame
        """

        self.frame_length = int(np.floor(duration * self.fs / 1000))  # frame length (in samples) s -> msec

        self.frame_list = self.split_into_frames(self.frame_length, overlap_rate)
        window_filter = lb.filters.get_window(window=window_type, Nx=self.frame_length, fftbins=False)
        self.windowed_frame_list = np.multiply(self.frame_list, window_filter)  # [np.multiply(frame, window_filter) for frame in frame_list] # windowing the frames

    def FFT(self):
        # return the spectrum magnitude
        frameWind_list = self.frame_list
        return [np.abs(np.fft.fft(frameWind)) for frameWind in frameWind_list]

    def Max_Freq(self):
        # return the frequency corrisponding to the maximum value of the spectrum magnitude
        fft_list = self.FFT()
        frequencies = np.fft.fftfreq(self.frame_length, 1/self.fs)
        return np.array([frequencies[np.argmax(fft**2)] for fft in fft_list])

    def Magnitude(self):
        # return the magnitude of the frame
        frameWind_list = self.frame_list
        return [np.abs(frameWind) for frameWind in frameWind_list]

    def ZCR(self):
        # return the zero crossing rate for each (overlapping) frame
        frameWind_list = self.frame_list #self.windowed_frame_list
        return [np.sum(abs(np.diff(np.sign(frameWind - np.mean(frameWind, axis=0))))) / (2 * self.frame_length) for frameWind in frameWind_list]

    def Energy(self):
        # return the energy for each (overlapping) frame
        frameWind_list = self.frame_list #self.windowed_frame_list #self.frame_list
        return [np.sum(frameWind ** 2, axis=0) for frameWind in frameWind_list]


    def MSE_Energy(self):
        # return the mean square error of the energy for each (overlapping) frame
        energy_list = self.Energy()
        return [np.sqrt(energy) for energy in energy_list]

    def HTN(self):
        # return the Harmonic-to-noise for each (overlapping) frame
        frameWind_list = self.windowed_frame_list
        return [signal.get_HNR(signal=frameWind, rate=self.fs) for frameWind in frameWind_list]

    def SFM(self):
        # return the Spectral Flatness Measure
        magnitude_list = self.FFT()
        #sfm_list =
        return [10 * np.log10(np.mean(np.log(frameWind**2)) / np.mean(frameWind**2)) for frameWind in magnitude_list]

    #def STN(self):
        # return the Signal-to-noise for each (overlapping) frame
        #frameWind_list = self.windowed_frame_list
        #return [for frameWind in frameWind_list]

    def MFCC(self):
        # return the MFCC coefficients for each (overlapping) frame
        frameWind_list = self.windowed_frame_list
        return [feature.mfcc(frameWind, self.fs) for frameWind in frameWind_list]  # 20 coeff per frame