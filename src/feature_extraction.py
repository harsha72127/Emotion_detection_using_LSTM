import librosa
import numpy as np

def extract_mfcc(file_path=None, audio_data=None, sr=22050, n_mfcc=40):
    """
    Either provide:
    - file_path (for dataset usage) OR
    - audio_data + sr (for real-time usage)

    Returns mean MFCC feature vector
    """

    if file_path:
        audio_data, sr = librosa.load(file_path, sr=sr)

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean
