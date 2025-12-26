

# 🎤 Real-Time Voice Emotion Detection using Deep Learning (LSTM)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Librosa](https://img.shields.io/badge/Audio-Librosa-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

A **real-time speech emotion recognition system** powered by **MFCC audio features + LSTM Neural Network**, featuring a **clean modular codebase**, **GUI using Streamlit**, and **support for live microphone recording**.

The system captures microphone input in real time, extracts audio features, predicts emotions using a trained deep learning model, and displays classified emotions along with probability scores.

---

## 📌 Key Highlights

✔ Real-time microphone-based emotion prediction
✔ Trained on **RAVDESS Emotional Speech Dataset**
✔ **MFCC Feature Extraction + LSTM Model**
✔ Highly modular ML pipeline
✔ Streamlit UI for end users
✔ Visual emotion probability distribution
✔ Model evaluation with confusion matrix
✔ Ready for deployment and portfolio showcase

---

## 😃 Emotions Detected

* Happy
* Sad
* Angry
* Fearful
* Neutral
* Calm
* Disgust
* Surprise

---

## 🧠 Model Overview

| Component         | Description               |
| ----------------- | ------------------------- |
| Input             | Raw voice waveform        |
| Feature Extractor | MFCC via Librosa          |
| Model             | LSTM RNN                  |
| Framework         | PyTorch                   |
| Output            | Emotion Class Probability |

---

## 📂 Project Structure

```
emotion_detection
│
├── notebooks/
│   ├── training.ipynb
│   └── real_time_detection.ipynb
│
├── src/
│   ├── dataset_loader.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── train.py
│   ├── realtime.py
│
├── saved_models/
│   └── model.pth
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── assets/
    ├── demo.png
    └── confusion_matrix.png
```

---

## 📥 Dataset

This project uses the **RAVDESS Emotional Speech Audio Dataset**

Download from Kaggle:

[https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

Place it in:

```
emotion_detection/data/ravdess/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Linux / Mac:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

If PyAudio fails (Windows):

```
pip install pipwin
pipwin install pyaudio
```

If microphone not working, also install:

```
pip install sounddevice
```

---

## 🏋️ Training The Model

Open the notebook:

```
notebooks/training.ipynb
```

It will:

✔ Load dataset
✔ Extract MFCC features
✔ Train LSTM model
✔ Evaluate performance
✔ Save trained model to:

```
saved_models/model.pth
```

---

## 🎧 Real-Time Voice Detection (Notebook)

To test real-time emotions inside Jupyter:

```
notebooks/real_time_detection.ipynb
```

---

## 🖥️ Streamlit Application (Main UI)

Run:

```
streamlit run app.py
```

✔ Select recording duration
✔ Select sampling rate (default recommended: 22050 Hz)
✔ Press "Record & Predict Emotion"
✔ Speak
✔ View prediction & probability graph

---

## 📊 Model Evaluation

Confusion Matrix:

```
assets/confusion_matrix.png
```

Application Demo:

```
assets/demo.png
```

---

## 🧰 Tech Stack

* Python
* PyTorch
* Librosa
* NumPy
* Scikit-Learn
* Streamlit
* SoundDevice / PyAudio

---

## 🚀 Deployment Options

You can deploy using:

* Streamlit Cloud
* HuggingFace Spaces
* Local Desktop App
* Flask Backend + React UI
* Docker Container

---

## ❗ Troubleshooting

### 1️⃣ Librosa Warning: Empty Filters

If you see:

```
Empty filters detected in mel frequency basis
```

Fix:

* Ensure sampling rate = 22050
* Set `fmax = sr // 2`
* Use `n_mels = 40 or 64`

---

### 2️⃣ `train_test_split: n_samples = 0`

Means dataset failed to load or features empty.

Check:

```
print(len(X), len(y))
```

Ensure audio is valid & features extracted correctly.

---

### 3️⃣ Microphone Not Working

Install:

```
pip install sounddevice
pip install pyaudio
```

Run as administrator if needed.

---

### 4️⃣ Very Low Accuracy?

* Train longer
* Increase MFCC features
* Add noise handling
* Normalize audio
* Use more training samples

---

## 🧪 Future Enhancements

* CNN + LSTM Hybrid
* Attention Mechanism
* Noise Resistant Training
* Multi-language support
* Mobile / Desktop App
* Cloud Deployment
* Real-time streaming support

---

## 👤 Author

**Chedalla Harsha**

---

## 📜 License

This project is licensed under **MIT License**
Free to use, modify and distribute.
