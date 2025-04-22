import librosa
import numpy as np

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)

    features = np.concatenate((
        np.mean(mfccs.T, axis=0),
        np.mean(spectral_contrast.T, axis=0),
        np.mean(chroma.T, axis=0)
    ))

    return features
