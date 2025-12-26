import streamlit as st
import sounddevice as sd
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.model import LSTMEmotionModel
from src.feature_extraction import extract_mfcc

# ---------------------------------------
# Page Setup
# ---------------------------------------
st.set_page_config(page_title="Emotion Detection AI", layout="centered")
st.title("🎤 Real-Time Voice Emotion Detection")
st.caption("Detect human emotions from speech using Deep Learning (LSTM)")

# ---------------------------------------
# Load Model
# ---------------------------------------
MODEL_PATH = "saved_models/model.pth"
EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

model = LSTMEmotionModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ---------------------------------------
# UI Controls
# ---------------------------------------
st.subheader("🎛 Recording Settings")
duration = st.slider("Recording Duration (seconds)", 1, 30, 2, 1)
sr = st.selectbox("Sampling Rate", [16000, 22050, 44100], index=1)

# ---------------------------------------
# Audio Recorder
# ---------------------------------------
def record_audio(duration, sr):
    st.info("🎙 Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    st.success("✔ Recording complete")
    return audio.flatten()

# ---------------------------------------
# Prediction
# ---------------------------------------
def predict(audio):
    mfcc = extract_mfcc(audio_data=audio, sr=sr)
    inp = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(inp)

    probs = torch.softmax(output, dim=1).numpy()[0]
    emotion_idx = np.argmax(probs)
    
    return EMOTIONS[emotion_idx], probs

# ---------------------------------------
# Main Button
# ---------------------------------------
if st.button("🎧 Record & Predict Emotion", use_container_width=True):
    try:
        audio = record_audio(duration, sr)

        # Waveform
        st.subheader("📈 Audio Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)


        # Play Audio
        st.audio(audio, sample_rate=sr)

        # Prediction
        st.subheader("🧠 AI Prediction")
        with st.spinner("Analyzing emotion..."):
            emotion, probs = predict(audio)

        confidence = float(np.max(probs)) * 100
        st.success(f"Detected Emotion: **{emotion.upper()}** ({confidence:.2f}% confidence)")

        # Probability Chart
        st.subheader("📊 Emotion Probability Distribution")
        prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        st.bar_chart(prob_dict)

    except Exception as e:
        st.error("❌ Something went wrong. Check microphone permission or try again.")
        st.exception(e)

# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("---")
st.caption("Developed by **Chedalla Harsha** | Powered by LSTM + PyTorch + Streamlit")
