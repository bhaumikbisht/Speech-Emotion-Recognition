import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# ==========================================================
# LOAD TRAINED MODEL & LABEL ENCODER
# ==========================================================

model = load_model("models/emotion_model.keras")
label_encoder = joblib.load("emotion_label_encoder.pkl")

# ==========================================================
# SELECT AUDIO FILE
# ==========================================================

# Put your test .wav file inside the test_audio folder  
audio_file = "test_audio/03-01-02-02-01-01-01.wav"

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return np.expand_dims(mfcc, axis=0)

# ==========================================================
# CHECK FILE
# ==========================================================

if not os.path.exists(audio_file):
    print("\n❌ Audio file not found!")
    print(f"Expected: {audio_file}")
    print("\nPut a .wav file inside the 'test_audio' folder.")
    exit()

# ==========================================================
# PREDICT
# ==========================================================

features = extract_features(audio_file)

prediction = model.predict(features, verbose=0)

predicted_index = np.argmax(prediction)

emotion = label_encoder.inverse_transform([predicted_index])[0]

confidence = np.max(prediction) * 100

# ==========================================================
# RESULTS
# ==========================================================

print("\n==============================================")
print("     EMOTION RECOGNITION RESULT")
print("==============================================")
print(f"Audio File        : {os.path.basename(audio_file)}")
print(f"Predicted Emotion : {emotion.upper()}")
print(f"Confidence        : {confidence:.2f}%")
print("==============================================")