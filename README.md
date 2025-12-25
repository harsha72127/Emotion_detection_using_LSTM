# Emotion_detection_using_LSTM

# 🎤 Real-Time Emotion Detection from Voice using LSTM (RNN)

This project implements a **real-time human emotion detection system from speech audio** using **Deep Learning (LSTM-based RNN)**. The model is trained on the **RAVDESS Emotional Speech Dataset** and performs live emotion recognition using microphone input.

It demonstrates end-to-end capabilities in:

* Audio Signal Processing
* Feature Engineering (MFCCs)
* Deep Learning with PyTorch
* Real-Time Inference from Microphone Input
* Deployment-ready structure

---

## 🚀 Features

✔ Recognizes emotions from voice in real time
✔ Uses **LSTM (RNN)** for sequential modeling
✔ Extracts **MFCC features** using Librosa
✔ Trained on **RAVDESS Dataset**
✔ Achieves strong accuracy on test samples
✔ Modular & clean notebook structure
✔ Works offline after setup

---

## 🎯 Detected Emotions

The system classifies speech into multiple emotions including:

* Happy
* Sad
* Angry
* Fearful
* Calm
* Neutral
* Disgust
* Surprise

---

## 🧠 Model Architecture

* Input: MFCC features (time-series audio representation)
* Model: LSTM-based RNN
* Framework: PyTorch
* Output: Softmax emotion classification

---

## 📂 Project Structure

```
emotion_detection/
│
├── data/                     # RAVDESS dataset (not uploaded to repo)
├── training.ipynb            # Model training notebook
├── Real_time_detection.ipynb # Live microphone-based prediction
├── model.pth                 # Trained model weights
├── requirements.txt
└── README.md
```

---

## 📥 Dataset

This project uses the **RAVDESS Emotional Speech Audio Dataset**.

Download from Kaggle:
[https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

Extract it to:

```
emotion_detection/data/ravdess/
```

---

## ⚙️ Installation

### 1️⃣ Create Environment

```
python -m venv venv
```

Activate:

Windows:

```
venv\Scripts\activate
```

Linux/Mac:

```
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

If PyAudio fails on Windows:

```
pip install pipwin
pipwin install pyaudio
```

---

## 🏋️‍♂️ Training

Open:

```
training.ipynb
```

Run all cells to:
✔ Load dataset
✔ Extract MFCC
✔ Train LSTM Model
✔ Save `model.pth`

---

## 🔴 Real-Time Emotion Detection

Open:

```
Real_time_detection.ipynb
```

Run all cells.
Speak into your mic when prompted.
You will see:

```
Recording...
Predicted Emotion: Angry
```

---

## 📊 Results

* Successfully detects emotions from voice
* Good performance across most classes
* Demonstrates practicality of audio-based affect recognition

(You can add accuracy screenshots or a confusion matrix here later.)

---

## 🧰 Tech Stack

* Python
* PyTorch
* Librosa
* NumPy
* SoundDevice / PyAudio
* Scikit-learn
* Jupyter Notebook

---

## 🚀 Future Enhancements

Planned improvements:

* Streamlit / Web UI
* Mobile deployment
* Support multilingual datasets
* CNN + Attention architectures
* Noise-robust training

---

## 👤 Author

**Chedalla Harsha**

---

## 📜 License

This project is licensed under the MIT License.



* Add badges (Stars, License, Python Version, etc.)
* Add screenshots / GIF demo section
* Write a strong “Project Highlights for Resume”
* Make README more recruiter-focused

Tell me if you want a **simple**, **premium portfolio**, or **research style** README.
