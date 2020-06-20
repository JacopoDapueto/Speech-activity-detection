import librosa as lb
from librosa import feature
from Signal_Analysis.features import signal
import numpy as np
import librosa


def nextpow2(x):
    return np.ceil(np.log2(abs(x)))

class Frame():

    def __init__(self, data, fs, frame_duration, overlap_rate):
        self.data=np.ascontiguousarray(data[:,0], dtype=np.float32) if data.shape == (len(data), 2) else data # dealing with multichannel audio, if any
        self.fs = fs
        self.windowed_frames(duration=frame_duration, overlap_rate= overlap_rate, window_type="hamm")

    # SHORT TIME ANALISYS
    def split_into_frames(self, frame_length, overlap_rate):
        # The audio is divided into frames
        shift = 1.0 - overlap_rate
        self.frame_shift = round(frame_length * shift)  # shift length in samples: hop_length
        frame_list = lb.util.frame(x=self.data, frame_length=self.frame_length, hop_length=self.frame_shift, axis=0)  # matrix: each row represents a frame
        return frame_list

    def windowed_frames(self, duration=32, overlap_rate=0.0, window_type="hamming"):
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

    def getNumFrames(self):
        # return the number of frames
        return len(self.frame_list)

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
        return [np.linalg.norm(frameWind, ord=1) for frameWind in frameWind_list]

    def ZCR(self):
        # return the zero crossing rate for each frame
        frameWind_list = self.frame_list
        return [np.sum(abs(np.diff(np.sign(frameWind - np.mean(frameWind, axis=0))))) / (2 * self.frame_length) for frameWind in frameWind_list]

    def Energy(self):
        # return the energy for each frame
        frameWind_list = self.windowed_frame_list #self.frame_list
        return [np.sum(frameWind ** 2, axis=0) for frameWind in frameWind_list]

    def MSE_Energy(self):
        # return the mean square error of the energy for each frame
        energy_list = self.Energy()
        return [np.sqrt(energy) for energy in energy_list]

    def HTN(self):
        # return the Harmonic-to-noise for each frame
        frameWind_list = self.windowed_frame_list
        return [signal.get_HNR(signal=frameWind, rate=self.fs) for frameWind in frameWind_list]

    def SFM(self):
        # return the Spectral Flatness Measure
        magnitude_list = self.FFT()
        sfm_list = [np.exp(np.mean(np.log(frameWind**2))) / np.mean(frameWind**2) for frameWind in magnitude_list]
        return [10 * np.log10(sfm) for sfm in sfm_list]

    def MFCC(self):
        n_fft = int(2 ** nextpow2(self.frame_length))
        mfcc_frames = librosa.feature.mfcc(y=self.data, sr=self.fs, n_mfcc=13, hop_length=self.frame_shift, htk=False, win_length=self.frame_shift, n_fft=n_fft).T
        diff = mfcc_frames.shape[0] - len(self.frame_list) # if there are more frames, those are removed
        return mfcc_frames[:-diff]


