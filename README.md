# Speech-Emotion-Recognition
# 🎙️ Emotion Recognition from Speech using Deep Learning

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=250&color=0:0F172A,50:2563EB,100:06B6D4&text=Emotion%20Recognition%20From%20Speech&fontAlignY=38&fontSize=40&fontColor=FFFFFF&animation=fadeIn"/>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 📌 Project Overview

This project is a **Speech Emotion Recognition (SER)** system developed using **Deep Learning**. It analyzes human speech and predicts the speaker's emotional state from audio recordings.

The model is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and classifies speech into eight different emotion categories using **MFCC (Mel-Frequency Cepstral Coefficients)** as audio features.

---

# ✨ Features

- 🎤 Speech Emotion Recognition
- 🎵 MFCC Feature Extraction using Librosa
- 🧠 Deep Learning Neural Network (TensorFlow/Keras)
- 📊 Train/Test Dataset Split
- 💾 Model Saving and Loading
- ⚡ Real-Time Emotion Prediction from WAV Files
- 📈 Clean and Modular Python Code

---

# 😊 Supported Emotions

| Emotion | Label |
|----------|-------|
| 😐 Neutral | 01 |
| 😌 Calm | 02 |
| 😀 Happy | 03 |
| 😢 Sad | 04 |
| 😠 Angry | 05 |
| 😨 Fear | 06 |
| 🤢 Disgust | 07 |
| 😲 Surprise | 08 |

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Pandas

---

# 📂 Project Structure

```text
Emotion_Recognition_From_Speech/
│
├── dataset/
│   ├── Actor_01
│   ├── Actor_02
│   └── ...
│
├── models/
│   └── emotion_model.keras
│
├── test_audio/
│   └── sample.wav
│
├── feature_extraction.py
├── train_model.py
├── predict_emotion.py
├── emotion_label_encoder.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Emotion_Recognition_From_Speech.git
```

Move into the project

```bash
cd Emotion_Recognition_From_Speech
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Step 1 — Extract Features

```bash
python feature_extraction.py
```

---

## Step 2 — Train the Model

```bash
python train_model.py
```

---

## Step 3 — Predict Emotion

Place your `.wav` file inside the `test_audio` folder and update the file path in `predict_emotion.py`.

Run:

```bash
python predict_emotion.py
```

---

# 📊 Example Output

```text
==============================================
     EMOTION RECOGNITION RESULT
==============================================
Audio File        : 03-01-02-02-01-01-01.wav
Predicted Emotion : CALM
Confidence        : 87.63%
==============================================
```

---

# 🔍 How It Works

```
Speech Audio
      │
      ▼
Feature Extraction (MFCC)
      │
      ▼
Data Preprocessing
      │
      ▼
Deep Learning Model
      │
      ▼
Emotion Prediction
```

---

# 📈 Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

The dataset contains professionally recorded emotional speech from multiple actors covering eight different emotions.

---

# 💡 Future Improvements

- 🎙️ Live Microphone Emotion Detection
- 🌐 Web Application using Flask or Streamlit
- 📱 Mobile Deployment
- 📊 Attention-based Deep Learning Models
- 🎯 Higher Prediction Accuracy

---

# 👨‍💻 Author

**Bhaumik Bisht**

GitHub: https://github.com/bhaumikbisht

LinkedIn: *(Add your LinkedIn profile here)*

---

# ⭐ If you found this project helpful

Please consider giving this repository a **Star ⭐**.

It helps support the project and encourages future development.
