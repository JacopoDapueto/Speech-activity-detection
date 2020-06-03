import os
import numpy as np
import textgrids
from Utils.Classes import Classes

class ReadTextGrid():
     # this class reads the annotations taken from Praat in the format

    def __init__(self, root, frameDuration):
        self.root = root
        self.frameDuration = frameDuration # in msec
        self.classes = Classes()

    def getLabels(self, fileName, num_frames, useRoot=True):
        """
        :param fileName: path (from root) of the text file .textGrid
        :param num_frame: number of necessary frames
        :return : return frame by frame SPEECH/NONSPEECH labels
        """
        if useRoot:
            dir_file = os.path.join(self.root, fileName)
        else:
            dir_file = fileName
        return self.loadAndRead(dir_file, num_frames)

    def loadAndRead(self, dir_file, num_frames):
        label_list = self.readFile(dir_file)
        #print(len(label_list) - num_frames)
        if (len(label_list) - num_frames) > 3:
            raise Exception("The number of frames are too different")
        else:
            for _ in range(len(label_list) - num_frames):
                label_list.pop(-1)  # there could be more due to approximation, it can be removed
        return np.array(label_list).reshape((len(label_list), 1)), len(label_list)  # the number of labels can be less than the number of wanted frames

    def readFile(self, path):
        # return the list of SPEECH/NONSPEECH labels
        labeled_list  = []
        grid = textgrids.TextGrid(path)

        for interval in grid['silences']:
            label = int(interval.text)

            dur = interval.dur
            dur_msec = dur * 1000 # sec -> msec
            num_frames = int(round(dur_msec /self.frameDuration))

            for i in range(num_frames):
                labeled_list.append(self.classes.getLabel(label))

        return labeled_list