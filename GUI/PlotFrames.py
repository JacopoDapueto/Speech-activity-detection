from SpeechDetection.SpeechDetection import SpeechDetector
from SpeechDetection.Classes import Classes

import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import numpy as np
from librosa import load
import librosa as lb

class PlotFrames():

    def __init__(self, data, fs, frame_duration, overlap_rate, labeled_frames):
        self.data = data
        self.fs = fs
        self.labeled_frames = labeled_frames
        self.frame_length = int(np.floor(frame_duration * self.fs / 1000))
        self.overlap_rate = overlap_rate

    def plot_segments(self, t, frame_shift):
        t_frame_start = None
        t_frame_end = None
        for i, frame_labeled in enumerate(self.labeled_frames):
            idx = i * frame_shift
            t_frame = t[idx:idx + self.frame_length]

            if (frame_labeled == Classes.SPEECH):
                plt.axvspan(xmin= t[idx], xmax=t[idx + self.frame_length], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='g', label='Speech')

    def plot_signal_and_segments(self):
        Ns = len(self.data) # number of sample
        Ts = 1 / fs  # sampling period
        t = np.arange(Ns) * 1000 * Ts # time axis
        plt.plot(t, iss.data)

        shift = 1.0 - self.overlap_rate
        frame_shift = round(self.frame_length * shift)
        self.plot_segments(t, frame_shift)
        plt.legend(['Signal', 'Speech'])
        plt.show()


#fs, data = wf.read('FSA1.WAV')
fs, data = wf.read('Example_M.wav')
#data, fs = load(lb.util.example_audio_file())

data = np.array(data, dtype=np.float)

frame_duration = 10
overlap_rate = 0.0
iss = SpeechDetector(data, fs, frame_duration, overlap_rate)
labeled_frames, frame_list = iss.isSpeech()
pf = PlotFrames(data, fs, frame_duration, overlap_rate, labeled_frames)
pf.plot_signal_and_segments()




