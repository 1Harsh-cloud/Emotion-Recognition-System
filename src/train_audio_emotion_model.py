import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# Path to MFCC features
FEATURE_DIR = '../data/audio_features/'

# Map RAVDESS emotion codes to labels
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

X = []
y = []

# Load features and labels
for file in os.listdir(FEATURE_DIR):
    if file.endswith('.npy'):
        X.append(np.load(os.path.join(FEATURE_DIR, file)))
        # RAVDESS filename: '03-01-05-01-01-01-01.wav' (emotion is 3rd part)
        emotion_code = file.split('-')[2]
        y.append(emotion_map.get(emotion_code, 'unknown'))

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Build a simple dense neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[es])

# Save model and label encoder
model.save('audio_emotion_model.h5')
np.save('audio_emotion_labels.npy', le.classes_)
print('Model and labels saved!') 