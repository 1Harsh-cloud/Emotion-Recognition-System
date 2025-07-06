# Emotion Recognition System Using Cross-Modal Translation

## Overview

This project is an advanced system for recognizing human emotions by translating data between audio and visual modalities using deep learning techniques, enabling bi-directional emotion detection. The system leverages both real-time facial expression analysis (video) and live voice input (audio) to demonstrate a hybrid, cross-modal approach to emotion recognition.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Technologies Used](#technologies-used)
3. [Dependency Usage](#dependency-usage)
4. [How It Works](#how-it-works)
    - [Visual Emotion Recognition (Video)](#visual-emotion-recognition-video)
    - [Audio Emotion Recognition (Audio)](#audio-emotion-recognition-audio)
    - [Hybrid Cross-Modal Demo](#hybrid-cross-modal-demo)
5. [How to Run](#how-to-run)
6. [Key Features](#key-features)
7. [Limitations & Future Work](#limitations--future-work)
8. [References](#references)

---

## Project Structure

```
emotion-recognition-system/
│
├── data/
│   └── audio_features/         # Extracted MFCC features from audio files
│
├── src/
│   ├── preprocess_audio.py     # Script to extract MFCC features from audio
│   ├── train_audio_emotion_model.py # Script to train audio emotion model (optional)
│   ├── webcam_emotion_demo.py  # Real-time facial emotion recognition
│   ├── live_audio_emotion_demo.py # Live audio demo (placeholder/random)
│   ├── hybrid_emotion_demo.py  # Combined video+audio demo (cross-modal)
│   └── fer2013_mini_XCEPTION.102-0.66.hdf5 # Pre-trained video emotion model
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow/Keras:** Deep learning framework for building and running emotion recognition models.
- **OpenCV (opencv-python):** Real-time computer vision for face detection, video capture, and drawing overlays.
- **Librosa:** Audio processing and feature extraction (MFCCs) from speech.
- **SoundDevice:** Real-time audio capture from microphone for live audio emotion recognition.
- **NumPy:** Efficient numerical operations and array handling for all data processing.
- **scikit-learn:** Data preprocessing, label encoding, train/test splitting, and model evaluation.
- **Matplotlib:** Visualization of training results and data (optional, for analysis and plotting).
- **Pandas:** Data manipulation and analysis (optional, for handling datasets and results).
- **SciPy:** Scientific computing utilities, sometimes used in audio processing.
- **Flask:** (Optional) For building a web interface or API for the system.
- **Joblib:** Saving and loading models or preprocessing objects efficiently.

---

## Dependency Usage

| Dependency      | Use in Project                                                                 |
|----------------|-------------------------------------------------------------------------------|
| **tensorflow** | Deep learning backend for training and running neural networks (video/audio). |
| **keras**      | High-level API for building and training deep learning models.                |
| **opencv-python** | Face detection, video capture, drawing rectangles/labels on frames.        |
| **librosa**    | Extracting MFCC features from audio for emotion recognition.                  |
| **numpy**      | Array operations, data manipulation, feature storage.                         |
| **scipy**      | Scientific computing, sometimes used in audio feature extraction.             |
| **matplotlib** | Plotting training curves, data visualization (optional).                      |
| **pandas**     | Data manipulation, reading/writing datasets (optional).                       |
| **scikit-learn** | Label encoding, train/test split, evaluation metrics.                       |
| **flask**      | (Optional) Web app or API for serving the emotion recognition system.         |
| **joblib**     | Saving/loading models, encoders, or other objects efficiently.                |

---

## How It Works

### Visual Emotion Recognition (Video)

- Uses your webcam to capture live video frames.
- Detects faces in each frame using OpenCV's Haar Cascade classifier.
- Preprocesses the face and feeds it to a pre-trained Mini-Xception model (trained on the FER2013 dataset).
- Predicts one of seven emotions:  
  **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**
- Displays the predicted emotion label above your face in real time.

### Audio Emotion Recognition (Audio)

- Continuously listens to your microphone.
- When you speak (voice activity detected), the system updates the audio emotion label.
- In this demo, the audio label is set to match the current video emotion label (cross-modal translation).
- If you are silent, the audio label remains stable.

> **Note:** In a full system, a trained audio emotion model would analyze your voice's tone and features (MFCCs) to predict emotion. Here, we use a cross-modal placeholder for demonstration.

### Hybrid Cross-Modal Demo

- Both video and audio emotion labels are displayed on the webcam window.
- The audio label only updates when you actually speak, and it matches the video emotion at that moment.
- Demonstrates bi-directional, cross-modal emotion recognition logic.

---

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the hybrid demo:**
   ```sh
   python src/hybrid_emotion_demo.py
   ```

3. **Usage:**
   - Look at the webcam and make different facial expressions.
   - Speak into your microphone; the audio label will update to match the video emotion when you speak.
   - Press `q` to quit.

---

## Key Features

- **Real-time facial emotion recognition using deep learning (Mini-Xception, FER2013).**
- **Live audio input with voice activity detection.**
- **Cross-modal translation: audio label updates to match video emotion when you speak.**
- **Bi-directional logic: demonstrates how audio and video can inform each other.**
- **Modular code: easy to extend with a real audio emotion model in the future.**

---

## Limitations & Future Work

- **Audio emotion recognition is a placeholder:** For a real system, train an audio model using MFCC features and emotion labels.
- **Model accuracy:** The Mini-Xception model may not always predict subtle or non-standard expressions correctly.
- **Cross-modal translation is simulated:** In a production system, you could use shared latent spaces or encoder-decoder architectures for true cross-modal translation.
- **Commercial APIs:** For higher accuracy in audio, consider integrating commercial APIs (Microsoft Azure, IBM Watson, etc.).

---

## References

- [Mini-Xception Model (oarriaga/face_classification)](https://github.com/oarriaga/face_classification)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

---

## Acknowledgements

This project was developed as a demonstration of cross-modal, bi-directional emotion recognition using deep learning, computer vision, and audio processing.  
**Author:** Harsh Jain


