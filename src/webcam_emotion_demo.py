import cv2
import numpy as np
from keras.models import load_model

# Emotion labels for FER2013
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained Mini-Xception model
model = load_model('src/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

VOLUME_THRESHOLD = 0.005  # or even lower, e.g., 0.001

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

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
        emotion = emotions[np.argmax(preds)]
        # Put the emotion label above the face
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Webcam Emotion Recognition (Mini-Xception)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Video emotion:", emotion)

cap.release()
cv2.destroyAllWindows() 