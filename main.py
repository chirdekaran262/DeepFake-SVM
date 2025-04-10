import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None
    
    # Extract MFCC features (original feature)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Feature 1: Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    # Feature 2: Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Combine all features
    combined_features = np.concatenate((mfccs_mean, spectral_contrast_mean, chroma_mean))
    
    return combined_features

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

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y


def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in training data:", unique_classes)
    print(f"Class distribution: {np.bincount(y)}")

    if len(unique_classes) < 2:
        raise ValueError("At least 2 classes are required to train")

    print("Size of X:", X.shape)
    print("Size of y:", y.shape)

    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Warning: At least one class has fewer than 2 samples")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
        print("Size of y_train:", y_train.shape)
        print("Size of y_test:", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Try different kernels to find the best one
    kernels = ['linear', 'rbf', 'poly']
    best_accuracy = 0
    best_model = None
    best_kernel = None

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        
        for kernel in kernels:
            svm_classifier = SVC(kernel=kernel, random_state=42, probability=True)
            svm_classifier.fit(X_train_scaled, y_train)
            
            y_pred = svm_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Kernel: {kernel}, Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = svm_classifier
                best_kernel = kernel
        
        print(f"\nBest kernel: {best_kernel} with accuracy: {best_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, best_model.predict(X_test_scaled)))
        
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, best_model.predict(X_test_scaled))
        print(conf_matrix)
    else:
        print("Insufficient samples for stratified splitting. Training on all available data.")
        best_model = SVC(kernel='linear', random_state=42, probability=True)
        best_model.fit(X_train_scaled, y_train)

    # Save the trained model and scaler
    model_filename = "enhanced_svm_model.pkl"
    scaler_filename = "enhanced_scaler.pkl"
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")
    
    return best_model, scaler

def analyze_audio(input_audio_path, model_filename="enhanced_svm_model.pkl", scaler_filename="enhanced_scaler.pkl"):
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        print(f"Error: Model files {model_filename} or {scaler_filename} not found.")
        return
    
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist .")
        return
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    features = extract_features(input_audio_path)

    if features is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = svm_classifier.predict(features_scaled)
        probability = svm_classifier.predict_proba(features_scaled)[0]
        
        result = "genuine" if prediction[0] == 0 else "deepfake"
        confidence = probability[prediction[0]] * 100
        
        print(f"The input audio is classified as {result} with {confidence:.2f}% confidence.")
        return {"result": result, "confidence": confidence}
    else:
        print("Error: Unable to process the input audio.")
        return None

def main():
    genuine_dir = input("Enter path to genuine audio directory: ")
    deepfake_dir = input("Enter path to deepfake audio directory: ")
    
    # Use default paths if not provided
    if not genuine_dir:
        genuine_dir = "real_audio"
    if not deepfake_dir:
        deepfake_dir = "deepfake_audio"
    
    if not os.path.exists(genuine_dir) or not os.path.exists(deepfake_dir):
        print("Error: One or both directories do not exist.")
        return

    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

    # Check if each class has enough samples
    if len(X_genuine) == 0 or len(X_deepfake) == 0:
        print("Error: One or both classes have no samples.")
        return
    
    X = np.vstack((X_genuine, X_deepfake))
    y = np.hstack((y_genuine, y_deepfake))

    model, scaler = train_model(X, y)
    
    while True:
        choice = input("\nDo you want to analyze an audio file? (y/n): ").lower()
        if choice != 'y':
            break
            
        user_input_file = input("Enter the path of the .wav file to analyze: ")
        analyze_audio(user_input_file)

if __name__ == "__main__":
    main()