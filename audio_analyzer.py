import os
import joblib
from feature_extraction import extract_features

def analyze_audio(audio_path, model_path="models/enhanced_svm_model.pkl", scaler_path="models/enhanced_scaler.pkl"):
    if not os.path.exists(audio_path) or not audio_path.lower().endswith(".wav"):
        print("Invalid audio file.")
        return

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler file missing.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features = extract_features(audio_path)
    if features is not None:
        scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled)
        prob = model.predict_proba(scaled)[0]
        label = "genuine" if prediction[0] == 0 else "deepfake"
        confidence = prob[prediction[0]] * 100
        print(f"Prediction: {label} ({confidence:.2f}%)")
        return {"result": label, "confidence": confidence}
    else:
        print("Could not extract features.")
        return None
