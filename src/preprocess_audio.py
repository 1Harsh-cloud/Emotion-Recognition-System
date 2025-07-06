import os
import numpy as np
import librosa
import joblib

DATA_DIR = '../data/'
OUTPUT_DIR = '../data/audio_features/'
SAMPLE_RATE = 22050
N_MFCC = 40

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=N_MFCC):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

def process_all_audio(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    mfcc = extract_mfcc(file_path)
                    # Save MFCC with same name as audio file
                    out_path = os.path.join(output_dir, file.replace('.wav', '.npy'))
                    np.save(out_path, mfcc)
                    print(f"Processed {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == '__main__':
    process_all_audio() 