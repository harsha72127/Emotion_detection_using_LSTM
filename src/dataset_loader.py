import os
import numpy as np
from .feature_extraction import extract_mfcc

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def load_ravdess_dataset(dataset_path):
    X, y = [], []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = EMOTION_MAP.get(emotion_code)

                fpath = os.path.join(root, file)
                features = extract_mfcc(file_path=fpath)

                X.append(features)
                y.append(emotion)

    return np.array(X), np.array(y)
