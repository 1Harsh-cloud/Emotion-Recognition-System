import cv2
import numpy as np
from keras.models import load_model
import sounddevice as sd
import threading
import librosa

# Emotion labels for FER2013
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained Mini-Xception model
model = load_model('src/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Audio settings
SAMPLE_RATE = 22050
FRAME_DURATION = 0.5  # seconds per audio frame
N_MFCC = 40
VOLUME_THRESHOLD = 0.01  # Adjust as needed for your mic

audio_emotion_label = '...'
latest_video_emotion = '...'

def audio_thread_func():
    global audio_emotion_label, latest_video_emotion
    while True:
        audio = sd.rec(int(FRAME_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        volume = np.sqrt(np.mean(audio**2))
        if volume > VOLUME_THRESHOLD:
            # User is speaking: update audio label to match video emotion
            audio_emotion_label = latest_video_emotion
        # else: keep the previous label

# Start audio thread
threading.Thread(target=audio_thread_func, daemon=True).start()

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    video_emotion = '...'
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Preprocess the face for the model
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)  # (64, 64, 1)
        face = np.expand_dims(face, axis=0)   # (1, 64, 64, 1)
        # Predict emotion
        preds = model.predict(face)
        video_emotion = emotions[np.argmax(preds)]
        # Put the video emotion label above the face
        cv2.putText(frame, f'Video: {video_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Update the latest video emotion for the audio thread
    latest_video_emotion = video_emotion

    # Show audio emotion label at the top left
    cv2.putText(frame, f'Audio: {audio_emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Hybrid Emotion Recognition Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 