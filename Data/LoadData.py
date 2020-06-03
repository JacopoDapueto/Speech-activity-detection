import os
import numpy as np
import scipy.io.wavfile as wf
from sklearn.model_selection import train_test_split

import Main
from SpeechDetection.Frame import Frame
from Utils.Utils import createSample
from Utils.Gender import Gender
from Data.ReadAnnotation import ReadTextGrid


FILELENGTH = 12

class LoadData():
    '''
    It load all the audio files with the corresponding annotations
    all the feature are created and then the dataset is divided into Train and Test dataset
    '''
    def __init__(self):
        self.root = str(__file__)[:-FILELENGTH] # folder containing the data
        self.gender_list = ['Male', 'Female']
        self.annotation_name = 'Annotation'
        self.audio_name = 'Audio'

    def getFilePerGender(self, path):
        file_list =  os.listdir(path)
        return [ os.path.join(path, file) for file in file_list]

    def getAnnotationPath(self):
        path = os.path.join(self.root, self.annotation_name)
        male, female = self.gender_list
        female_list_path = self.getFilePerGender(os.path.join(path, female))

        male_list_path = self.getFilePerGender(os.path.join(path, male))

        return female_list_path, male_list_path

    def getAudioPath(self):
        path = os.path.join(self.root, self.audio_name)
        male, female = self.gender_list

        path_female = os.path.join(path, female)
        female_list_path = self.getFilePerGender(path_female)

        path_male = os.path.join(path, male)
        male_list_path = self.getFilePerGender(path_male)

        return female_list_path, male_list_path

    def samplePerGender(self, audio_list, annotation_list, gender):

        if len(audio_list) != len(annotation_list):
            raise Exception("Each audio should have its own annotation file!")

        loadAnnotation = ReadTextGrid(os.path.join(self.root, self.annotation_name), Main.FRAMEDURATION)
        first = True

        for audio_path, annotation_path in zip(audio_list, annotation_list):
            fs, data = wf.read(audio_path)
            data = np.array(data, dtype=np.float)
            frames = Frame(data, fs, Main.FRAMEDURATION, Main.OVERLAPRATE)
            zcr_frames = frames.ZCR()
            energy_frames = frames.Energy()
            htn_frames = frames.HTN()
            #sfm_frames = frames.SFM()
            magnitude_frames = frames.Magnitude()
            mfcc_frames = frames.MFCC()

            labeled_frames, num_frame = loadAnnotation.getLabels(annotation_path, frames.getNumFrames(), useRoot=False)
            #print('---')
            #print(frames.getNumFrames() - num_frame, audio_path, annotation_path)
            #print(frames.getNumFrames() - num_frame)
            if frames.getNumFrames() - num_frame > 2:
                raise Exception("The number of frames are too different:")
            else:
                for _ in range(frames.getNumFrames() - num_frame):
                    # there are more due to approximation, it can be removed
                    zcr_frames.pop(-1)
                    energy_frames.pop(-1)
                    htn_frames.pop(-1)
                    #sfm_frames.pop(-1)
                    magnitude_frames.pop(-1)
                    mfcc_frames = np.delete(mfcc_frames, -1, axis=0)

            sample = createSample(zcr_frames, energy_frames, htn_frames, magnitude_frames, mfcc_frames, gender)
            if first:
                datasetX = sample.copy()
                datasetY = labeled_frames.copy()
                first  = False
            else:
                datasetX = np.vstack((datasetX, sample))
                datasetY = np.vstack((datasetY, labeled_frames))

        return datasetX, datasetY

    def loadAudioAndAnnotation(self):

        female_audio_list, male_audio_list = self.getAudioPath()
        female_annotation_list, male_annotation_list = self.getAnnotationPath()

        # female samples
        female_sample, female_y = self.samplePerGender(female_audio_list, female_annotation_list, Gender.FEMALE)

        # male samples
        male_sample, male_y = self.samplePerGender(male_audio_list, male_annotation_list, Gender.MALE)

        return np.vstack((female_sample, male_sample)), np.vstack((female_y, male_y))


    def getTrainAndTest(self):
        X, y= self.loadAudioAndAnnotation()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 16)
        return X_train, X_test, y_train, y_test
