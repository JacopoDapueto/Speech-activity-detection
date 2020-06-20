import tkinter as tk
from tkinter import font as tkfont
import tkinter.ttk as ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import wavfile
import numpy as np

from GUI.CONFIG import *
from GUI.PlotFrames import PlotFrames
from SpeechDetection.Frame import Frame
from SpeechDetection.SpeechDetection import SpeechDetector
from SpeechDetection.NeuralNetwork.Network import Network
from Data.ReadAnnotation import ReadTextGrid
import Data.Utils as Utils
import CONFIG

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def popup_message(msg):
    popup = tk.Tk()
    popup.resizable(RESIZABLE, RESIZABLE)
    popup.configure(bg=BACK_GROUND_COLOR)
    popup.geometry('300x150')
    popup.wm_title("ERROR!")
    font = tkfont.Font(family='arial black', size=20, weight="bold")
    label = tk.Label(master=popup, text=msg, font=font, bg=BACK_GROUND_COLOR)
    label.pack(pady=20)
    B1 = ttk.Button(popup, text="exit", command=popup.destroy)
    B1.pack(side='bottom', pady=25)


def plot_on_tab(figure, master):
    canvas = FigureCanvasTkAgg(figure=figure, master=master)
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, master)
    toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH)
    toolbar.update()


def create_frame_plot(tab):
    frame = tk.Frame(master=tab, height=HEIGHT_WINDOW - 55,
                     width=WIDTH_WINDOW - 215)
    frame.place(x=0, y=0)
    return frame


def update_frame_plot(frame, tab):
    frame.destroy()
    frame_ = create_frame_plot(tab=tab)
    return frame_


class PlotsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.file_path = ""
        self.annotation_path = ""
        self.bg = BACK_GROUND_COLOR
        self.font_label = self.title_font = tkfont.Font(family='Helvetica', size=10, weight="bold", slant="italic")

        self.network = Network()
        self.network.loadNetwork()

        # --------------------------------------------------------------------------------------------------

        self.upload_frame = tk.LabelFrame(master=self, text='Chose File Audio', font=self.font_label, height=90,
                                          width=130, borderwidth=2, relief='flat', highlightbackground="black",
                                          highlightcolor="black", highlightthickness=1,
                                          bg=LABEL_FRAME_COLOR).place(x=60, y=20)

        self.upload_annotation_frame = tk.LabelFrame(master=self, text='Chose Annotation', font=self.font_label, height=90,
                                          width=130, borderwidth=2, relief='flat', highlightbackground="black",
                                          highlightcolor="black", highlightthickness=1,
                                          bg=LABEL_FRAME_COLOR).place(x=60, y=110)

        self.accuracy_frame = tk.LabelFrame(master=self, text='Balanced accuracy', font=self.font_label, height=90,
                                          width=130, borderwidth=2, relief='flat', highlightbackground="black",
                                          highlightcolor="black", highlightthickness=1,
                                          bg=LABEL_FRAME_COLOR).place(x=60, y=400)

        self.upload_button = tk.Button(master=self, text='File Explorer',
                                       height=1, width=10, relief='groove', activebackground=BOTTON_ACTIVATION_COLOR,
                                       bg=BOTTON_COLOR, command=self.upload_file).place(x=80, y=45)

        self.upload_annotation_button = tk.Button(master=self, text='File Explorer',
                                       height=1, width=10, relief='groove', activebackground=BOTTON_ACTIVATION_COLOR,
                                       bg=BOTTON_COLOR, command=self.upload_annotation).place(x=80, y=135)

        self.tabControl = ttk.Notebook(master=self, height=HEIGHT_WINDOW - 50,
                                       width=WIDTH_WINDOW - 210)

        self.audio_var = tk.StringVar()
        self.annotation_var = tk.StringVar()
        self.accuracy_text_var = tk.StringVar()

        self.tab_classifier = ttk.Frame(self.tabControl)
        self.frame_plot_classifier = create_frame_plot(tab=self.tab_classifier)

        self.tab_ZCR = ttk.Frame(self.tabControl)
        self.frame_plot_ZCR = create_frame_plot(tab=self.tab_ZCR)

        self.tab_MAG = ttk.Frame(self.tabControl)
        self.frame_plot_MAG = create_frame_plot(tab=self.tab_MAG)

        self.tab_HTN = ttk.Frame(self.tabControl)
        self.frame_plot_HTN = create_frame_plot(tab=self.tab_HTN)

        self.tab_ENERGY = ttk.Frame(self.tabControl)
        self.frame_plot_ENERGY = create_frame_plot(tab=self.tab_ENERGY)


        self.tabControl.place(x=200, y=20)
        self.tabControl.add(self.tab_classifier, text='Speech/Nonspeech')
        self.tabControl.add(self.tab_ZCR, text='ZCR')
        self.tabControl.add(self.tab_MAG, text='MAGNITUDE')
        self.tabControl.add(self.tab_HTN, text='HTN')
        self.tabControl.add(self.tab_ENERGY, text='ENERGY')

        self.plot_button = tk.Button(master=self, text='Plot', activebackground=BOTTON_ACTIVATION_COLOR,
                                     height=1, width=10, relief='groove',
                                     bg=BOTTON_COLOR, command=self.plot_signal).place(x=80, y=205)

    def upload_file(self):
        self.annotation_path = ""
        self.annotation_var.set("")
        self.accuracy_text_var.set("")

        self.file_path = filedialog.askopenfilename(initialdir=Utils.DIR_LOCATION, title="Select Audio File", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        font = tkfont.Font(family='Helvetica', size=7)
        self.audio_text_label = tk.Message(master=self, relief='groove', width=100,
                                textvariable=self.audio_var, font=font).place(x=70, y=75)

        self.audio_var.set(str(self.file_path).split('/')[-1])
    def upload_annotation(self):
        self.annotation_path = filedialog.askopenfilename(initialdir=Utils.DIR_LOCATION, title="Select Annotation", filetypes=(("TextGrid files", "*.TextGrid"), ("all files", "*.*")))
        font = tkfont.Font(family='Helvetica', size=7)
        self.annotation_text_label = tk.Message(master=self, relief='groove', width=105,
                                textvariable=self.annotation_var, font=font).place(x=70, y=165)

        self.annotation_var.set(str(self.annotation_path).split('/')[-1])

    def show_accuracy(self, value):
        font = tkfont.Font(family='Helvetica', size=10)
        self.accuracy_text_label = tk.Message(master=self, relief='groove', width=105,
                                                textvariable=self.accuracy_text_var, font=font).place(x=70, y=440)

        self.accuracy_text_var.set(str(np.round(value, decimals=4) * 100) + "%")

    def plot_signal(self):
        self.frame_plot_classifier = update_frame_plot(self.frame_plot_classifier, tab=self.tab_classifier)
        self.frame_plot_ZCR = update_frame_plot(self.frame_plot_ZCR, tab=self.tab_ZCR)
        self.frame_plot_MAG = update_frame_plot(self.frame_plot_MAG, tab=self.tab_MAG)
        self.frame_plot_HTN = update_frame_plot(self.frame_plot_HTN, tab=self.tab_HTN)
        self.frame_plot_ENERGY = update_frame_plot(self.frame_plot_ENERGY, tab=self.tab_ENERGY)


        if self.file_path == "":
            popup_message("First of all upload the audio file!")
            return

        # read wav file
        fs, data = wavfile.read(self.file_path)
        data = np.array(data, dtype=np.float)
        frames = Frame(data, fs, CONFIG.FRAMEDURATION, CONFIG.OVERLAPRATE)
        classifier = SpeechDetector(data, fs, CONFIG.FRAMEDURATION, CONFIG.OVERLAPRATE, self.network)
        labeled_frames, _ = classifier.getLabels()

        if self.annotation_path != "":
            loadAnnotation = ReadTextGrid(Utils.DIR_LOCATION, CONFIG.FRAMEDURATION)
            try:
                labeled_ground_truth, num_ground_truth = loadAnnotation.getLabels(self.annotation_path, frames.getNumFrames(), useRoot=False)
            except Exception:
                popup_message("Wrong annotation file!")
                return
            plotAnnotation = PlotFrames(data, fs, CONFIG.FRAMEDURATION, CONFIG.OVERLAPRATE, labeled_ground_truth, "Ground truth")

            if frames.getNumFrames() - num_ground_truth > 4:
                popup_message("Wrong annotation file!")
            else:
                for _ in range(frames.getNumFrames() - num_ground_truth):
                    # there are more due to approximation, it can be removed
                    np.delete(labeled_frames, -1)

        plotframe = PlotFrames(data, fs, CONFIG.FRAMEDURATION, CONFIG.OVERLAPRATE, labeled_frames, "Predicted")

        figure = plt.Figure(figsize=(10, 7), dpi=85)
        figure.suptitle(str(self.file_path).split('/')[-1], fontsize=15)
        if self.annotation_path == "":
            ax = figure.add_subplot(111)
            ax.set_xlabel('time(milliseconds)')
        else:
            ax = figure.add_subplot(211)


        ax.set_ylabel('signal')
        plotframe.plot_signal_and_segments(ax)

        if self.annotation_path != "":
            self.show_accuracy(self.network.balance_accuracy(labeled_frames, labeled_ground_truth))
            ax = figure.add_subplot(212)
            ax.set_xlabel('time(milliseconds)')
            ax.set_ylabel('signal')
            plotAnnotation.plot_signal_and_segments(ax)
        plot_on_tab(figure=figure, master=self.frame_plot_classifier)

        figure2 = plt.Figure(figsize=(9, 5), dpi=90)
        ax2 = figure2.add_subplot(111)
        ax2.set_title('Short-Time Zero Crossing Rate', fontsize=20)
        ax2.set_xlabel('time')
        ax2.set_ylabel('zcr')
        plotframe.plot_feature(frames.ZCR(), ax2)
        plot_on_tab(figure=figure2, master=self.frame_plot_ZCR)

        figure3 = plt.Figure(figsize=(9, 5), dpi=90)
        ax3 = figure3.add_subplot(111)
        ax3.set_title('Short-Time Magnitude', fontsize=20)
        ax3.set_xlabel('time')
        ax3.set_ylabel('magnitude')
        plotframe.plot_feature(frames.Magnitude(), ax3)
        plot_on_tab(figure=figure3, master=self.frame_plot_MAG)

        figure4 = plt.Figure(figsize=(9, 5), dpi=90)
        ax4 = figure4.add_subplot(111)
        ax4.set_title('Short-Time Harmonic-To-Noise Ratio', fontsize=20)
        ax4.set_xlabel('time')
        ax4.set_ylabel('htn')
        plotframe.plot_feature(frames.HTN(), ax4)
        plot_on_tab(figure=figure4, master=self.frame_plot_HTN)

        figure5 = plt.Figure(figsize=(9, 5), dpi=90)
        ax5 = figure5.add_subplot(111)
        ax5.set_title('Short-Time Energy', fontsize=20)
        ax5.set_xlabel('time')
        ax5.set_ylabel('energy')
        plotframe.plot_feature(frames.Energy(), ax5)
        plot_on_tab(figure=figure5, master=self.frame_plot_ENERGY)


