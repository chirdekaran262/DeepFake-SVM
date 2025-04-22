from dataset_builder import create_dataset
from model_trainer import train_model
from audio_analyzer import analyze_audio
import numpy as np

def main():
    genuine_dir = input("Enter genuine audio folder (default: real_audio): ") or "real_audio"
    fake_dir = input("Enter deepfake audio folder (default: deepfake_audio): ") or "deepfake_audio"

    X_real, y_real = create_dataset(genuine_dir, 0)
    X_fake, y_fake = create_dataset(fake_dir, 1)

    if not X_real or not X_fake:
        print("Not enough data to train.")
        return

    X = np.vstack((X_real, X_fake))
    y = np.hstack((y_real, y_fake))

    model, scaler = train_model(X, y)

    while True:
        choice = input("Analyze audio? (y/n): ").lower()
        if choice != 'y':
            break
        file_path = input("Enter path to audio (.wav): ")
        analyze_audio(file_path)

if __name__ == "__main__":
    main()
