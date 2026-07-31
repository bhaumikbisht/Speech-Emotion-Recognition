import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Dataset path
DATASET_PATH = "dataset"

# Emotion mapping based on RAVDESS filenames
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


def extract_features(file_path):
    """
    Extract 40 MFCC features from an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print(f"Error reading {file_path}")
        print(e)
        return None


def load_dataset():

    X = []
    y = []

    for actor in os.listdir(DATASET_PATH):

        actor_path = os.path.join(DATASET_PATH, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                parts = file.split("-")

                emotion_code = parts[2]

                if emotion_code not in emotion_dict:
                    continue

                emotion = emotion_dict[emotion_code]

                file_path = os.path.join(actor_path, file)

                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(emotion)

    return np.array(X), np.array(y)


print("=" * 60)
print("Emotion Recognition From Speech")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y = load_dataset()

print(f"Total Samples : {len(X)}")

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(encoder, "emotion_label_encoder.pkl")

print("\nEmotion Classes")
print(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Split Successfully")

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nFeature Extraction Completed Successfully!")