import tkinter as tk
from tkinter import font as tkfont
from GUI.PlotsPage import PlotsPage
from GUI.CONFIG import *


class Gui(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('SPEECH ACTIVITY DETECTION')
        self.resizable(RESIZABLE, RESIZABLE)
        self.geometry(str(WIDTH_WINDOW) + 'x' + str(HEIGHT_WINDOW))
        self.configure(bg=BACK_GROUND_COLOR)
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.iconbitmap(default='GUI/icons/speech.ico')


        container = tk.Frame(self, bg=BACK_GROUND_COLOR, borderwidth=0, highlightbackground="black",
                             highlightcolor="black", highlightthickness=1)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        page_name = PlotsPage.__name__
        frame = PlotsPage(parent=container, controller=self)
        self.frames[page_name] = frame

        # put all of the pages in the same location;
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")
        frame.configure(bg=BACK_GROUND_COLOR)

        self.show_frame("PlotsPage")

    def show_frame(self, page_name):
        """Show a frame for the given page name"""
        frame = self.frames[page_name]
        frame.tkraise()

