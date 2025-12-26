import sounddevice as sd
import numpy as np
import torch
import librosa

from .model import LSTMEmotionModel
from .feature_extraction import extract_mfcc

MODEL_PATH = "saved_models/model.pth"

encoder_classes = [
    'neutral',
    'calm',
    'happy',
    'sad',
    'angry',
    'fearful',
    'disgust',
    'surprised'
]

model = LSTMEmotionModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def record_audio(duration=2, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Done.")
    return audio.flatten(), sr

def predict_emotion():
    audio, sr = record_audio()

    mfcc = extract_mfcc(audio_data=audio, sr=sr)
    inp = torch.tensor(mfcc, dtype=torch.float32)
    inp = inp.unsqueeze(0)

    output = model(inp)
    emotion_idx = torch.argmax(output).item()

    print("Predicted Emotion:", encoder_classes[emotion_idx])

if __name__ == "__main__":
    while True:
        predict_emotion()
