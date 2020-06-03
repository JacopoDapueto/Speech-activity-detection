from Utils.Classes import Classes
from Data.LoadData import LoadData
from SpeechDetection.NeuralNetwork.Network import Network
import matplotlib.pyplot as plt
import numpy as np

class PlotFrames():

    def __init__(self, data, fs, frame_duration, overlap_rate, labeled_frames, title_plot):
        self.data = data
        self.fs = fs
        self.labeled_frames = labeled_frames
        self.frame_length = int(np.floor(frame_duration * self.fs / 1000))
        self.overlap_rate = overlap_rate
        self.title_plot = title_plot

    def plot_segments(self, t, frame_shift):
        t_frame_start = None
        t_frame_end = None
        for i, frame_labeled in enumerate(self.labeled_frames):
            idx = i * frame_shift
            t_frame = t[idx:idx + self.frame_length]

            if (frame_labeled == Classes.SPEECH):
                plt.axvspan(xmin= t[idx], xmax=t[idx + self.frame_length-1], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='g', label='Speech')
            #else:
                #plt.axvspan(xmin=t[idx], xmax=t[idx + self.frame_length-1], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='r', label='Non-Speech')

    def plot_signal_and_segments(self):
        Ns = len(self.data) # number of sample
        Ts = 1 / self.fs  # sampling period
        t = np.arange(Ns) * 1000 * Ts # time axis
        plt.plot(t, self.data)

        shift = 1.0 - self.overlap_rate
        frame_shift = round(self.frame_length * shift)
        self.plot_segments(t, frame_shift)
        plt.legend(['Signal', 'Speech'])
        plt.title(self.title_plot)
        plt.show()

name = 'mic_F04_sx147'

#fs, data = wf.read(name + '.wav')

#data = np.array(data, dtype=np.float)

frame_duration = 30
overlap_rate = 0.0
#iss = SpeechDetector(data, fs, frame_duration, overlap_rate)
#labeled_frames, frame_list = iss.isSpeech()
#f=Frame(data, fs, frame_duration, overlap_rate)

#iss = ReadTextGrid("C:\\Users\\jacop\\PycharmProjects\\Speech-activity-detection\\Data\\Annotations\\Female", frame_duration, f.getNumFrames() )
#labeled_frames , num = iss.getLabels(name + ".TextGrid")


#print(len(labeled_frames),f.getNumFrames())
#pf = PlotFrames(data, fs, frame_duration, overlap_rate, labeled_frames, name)
#pf.plot_signal_and_segments()

