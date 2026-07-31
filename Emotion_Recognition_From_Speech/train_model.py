import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------

print("=" * 60)
print("Emotion Recognition Training")
print("=" * 60)

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

label_encoder = joblib.load("emotion_label_encoder.pkl")

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

num_classes = len(label_encoder.classes_)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# -----------------------------
# CNN Model
# -----------------------------

model = Sequential()

model.add(Conv1D(64, kernel_size=3, activation='relu',
                 input_shape=(40, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.keras")

print("\nModel saved successfully!")

# -----------------------------
# Evaluation
# -----------------------------

loss, accuracy = model.evaluate(X_test, y_test_cat)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=label_encoder.classes_
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Loss Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print("\nTraining Completed Successfully!")