from Utils.Classes import Classes
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np

class PlotFrames():

    def __init__(self, data, fs, frame_duration, overlap_rate, labeled_frames, title_plot):
        self.data = data
        self.fs = fs
        self.labeled_frames = labeled_frames.reshape((len(labeled_frames),))
        self.frame_length = int(np.floor(frame_duration * self.fs / 1000))
        self.overlap_rate = overlap_rate
        self.title_plot = title_plot

    def utilities(self):
        Ns = len(self.data)  # number of sample
        Ts = 1 / self.fs  # sampling period
        t = np.arange(Ns) * 1000 * Ts  # time axis

        shift = 1.0 - self.overlap_rate
        frame_shift = round(self.frame_length * shift)
        return t, frame_shift

    def plot_segments(self, t, frame_shift, p):

        for i, frame_labeled in enumerate(self.labeled_frames):
            idx = i * frame_shift
            if (frame_labeled == Classes.SPEECH):
                p.axvspan(xmin= t[idx], xmax=t[idx + self.frame_length-1], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='g', label='Speech')

    def plot_signal_and_segments(self, p):
        t, frame_shift = self.utilities()
        p.plot(t, self.data)

        self.plot_segments(t, frame_shift, p)
        p.legend(['Signal', 'Speech'])
        p.set_title(self.title_plot)

    def plot_feature(self,feature, p):
        Ts = 1 / self.fs
        time = []
        for i in range(len(feature)):
            time.append((i * self.frame_length) * Ts * 1000)
        p.plot(time, feature)

        t, frame_shift = self.utilities()
        self.plot_segments(t, frame_shift, p)
        p.legend(['feature', 'Speech prediction'])

    def plot_mfcc(self, mfcc, p):
        display.specshow(mfcc.T, x_axis='time', ax = p)
        p.set_title('MFCC')



