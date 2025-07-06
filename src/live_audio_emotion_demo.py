import sounddevice as sd
import numpy as np
import librosa
import random

# List of possible emotions (placeholder)
emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Disgust', 'Fear']

SAMPLE_RATE = 22050
DURATION = 2  # seconds per segment
N_MFCC = 40

print("Speak into the microphone. Press Ctrl+C to stop.")

try:
    while True:
        print("\nListening...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        # Extract MFCC features (not used for prediction in placeholder)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        # Randomly select an emotion (placeholder)
        emotion = random.choice(emotions)
        print(f"Predicted emotion: {emotion}")
except KeyboardInterrupt:
    print("\nStopped.") 