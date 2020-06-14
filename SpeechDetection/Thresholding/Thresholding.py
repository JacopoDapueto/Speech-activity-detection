import numpy as np
from Utils.Classes import Classes

class Thresholding():

    def __init__(self, frame):
        self.Frame = frame

    def refine_label(self, labeled_frames, label_to_be_replaced, label_used_to_replace, num_frames):
        idx_start = 0  # start of speech/non-speech segment
        idx_end = 0  # end of speech/non-speech segment

        idx_range = range(len(labeled_frames))
        for idx in idx_range:
            frame = labeled_frames[idx]
            if frame == label_to_be_replaced:  # is the label to refined
                if idx_end == 0:  # is the start of the segments
                    idx_start = idx
            else:
                idx_end = idx - 1  # the end of the segment
                if idx_end - idx_start < num_frames:  # to be ignored
                    for idx_sub in range(idx_start, idx_end + 1):
                        labeled_frames[idx_sub] = label_used_to_replace
                    idx_end = 0

        return labeled_frames

    def refining(self, labeled_frames):

        refined_labels = np.copy(labeled_frames)

        # Ignore silence run less than 10 successive frames
        refined_labels = self.refine_label(refined_labels, Classes.NONSPEECH, Classes.SPEECH, 20)

        # Ignore speech run less than 5 successive frames
        refined_labels = self.refine_label(refined_labels, Classes.SPEECH, Classes.NONSPEECH, 5)

        return refined_labels

    def thresholding(self):
        # initialization
        frame_list = self.Frames.frame_list
        sfm_list = self.Frames.SFM()
        energy_list = self.Frames.Energy()
        max_freq_list = self.Frames.Max_Freq()

        init_frames = 100//self.frame_duration # first 100ms
        threshold_sfm = 5
        init_threshold_energy = np.mean(energy_list[:init_frames]) #40 # mean to estimate the noise
        threshold_freq = 185

        labeled_frames = []

        buffer_dim = 10 #  Supposing that some of the first 30 frames are silence
        min_energy = min(energy_list[:buffer_dim]) # assuming there are at least 30 frames
        min_sfm = min(sfm_list[:buffer_dim]) # assuming there are at least 30 frames
        min_freq = min(max_freq_list[:buffer_dim])

        th_energy = 0
        silence_counter = 0

        for i, (frameWind, sfm, energy, max_freq) in enumerate(zip(frame_list, sfm_list, energy_list, max_freq_list)):
            th_energy = init_threshold_energy * np.log(min_energy)
            count = 0
            if (energy - min_energy) >= th_energy:
                count += 1

            if (sfm - min_sfm) >= threshold_sfm:
                count += 1

            if (max_freq - min_freq) >= threshold_freq:
                count += 1

            if count > 1:
                # is speech
                labeled_frames.append(Classes.SPEECH)
            else:
                # is non-speech
                silence_counter += 1
                labeled_frames.append(Classes.NONSPEECH)
                min_energy = ((silence_counter * min_energy) + energy)/(silence_counter + 1)
                #th_energy = init_threshold_energy * np.log(min_energy) # threshold updated

        return labeled_frames, frame_list