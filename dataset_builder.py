import os
import glob
from feature_extraction import extract_features

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    for audio_path in audio_files:
        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")
    print(f"Loaded {len(X)} samples from {directory}")
    return X, y
