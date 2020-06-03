import numpy as np


def createSample(zcr_frames, energy_frames, htn_frames, magnitude_frames, mfcc_frames, gender):
    rows = mfcc_frames.shape[0]

    zcr_frames= np.array(zcr_frames).reshape((rows, 1))
    energy_frames = np.array(energy_frames).reshape((rows, 1))
    htn_frames = np.array(htn_frames).reshape((rows, 1))
    #sfm_frames = np.array(sfm_frames).reshape((rows, 1))
    magnitude_frames = np.array(magnitude_frames).reshape((rows, 1))
    gender_frame = np.full(shape = (rows, 1), fill_value=gender)

    samples = np.hstack((mfcc_frames, zcr_frames, energy_frames,htn_frames, magnitude_frames, gender_frame))
    return samples