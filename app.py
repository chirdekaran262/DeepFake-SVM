import gradio as gr
import os
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
UPLOAD_FOLDER = 'uploads'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
try:
    model = joblib.load("enhanced_svm_model.pkl")
    scaler = joblib.load("enhanced_scaler.pkl")
    logger.info("Enhanced model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading enhanced model files: {e}")
    try:
        # Fallback to original model
        model = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("Fallback to original model successful")
    except Exception as e2:
        logger.error(f"Error loading fallback model files: {e2}")
        model = None
        scaler = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Extract Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Extract Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Combine all features
        combined_features = np.concatenate((mfccs_mean, spectral_contrast_mean, chroma_mean))
        
        return combined_features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        logger.error(traceback.format_exc())
        return None

def analyze_audio(audio_file):
    try:
        if audio_file is None:
            error_msg = "No file uploaded"
            return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
        
        if isinstance(audio_file, str):
            if not os.path.exists(audio_file):
                error_msg = f"File not found: {audio_file}"
                return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
            
            file_ext = os.path.splitext(audio_file)[1].lower().replace('.', '')
            if file_ext not in ALLOWED_EXTENSIONS:
                error_msg = f"Invalid file format: {file_ext}. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
                return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
            temp_path = audio_file
        else:
            try:
                if hasattr(audio_file, 'name'):
                    if not allowed_file(audio_file.name):
                        error_msg = f"Invalid file format. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
                        return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
                    
                    # Save uploaded file temporarily
                    temp_path = os.path.join(UPLOAD_FOLDER, os.path.basename(audio_file.name))
                    with open(temp_path, "wb") as f:
                        f.write(audio_file)
                else:
                    if isinstance(audio_file, tuple) and len(audio_file) >= 1:
                        temp_path = audio_file[0]
                    else:
                        error_msg = "Invalid audio format"
                        return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
        
        if model is None or scaler is None:
            error_msg = "Server configuration error. Models not loaded properly."
            return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
        
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            error_msg = "Could not extract features from the audio"
            return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"
        
        # Make prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction] * 100
            logger.info(f"Using predict_proba for confidence: {confidence}")
        else:
            confidence_score = np.max(np.abs(model.decision_function(features_scaled)))
            confidence = min(max(confidence_score * 50, 50), 99)
            logger.info(f"Using decision_function for confidence: {confidence}")
        
        label = "Real" if prediction == 0 else "Fake"
        confidence_text = f"{confidence:.2f}%"
        logger.info(f"Prediction: {label}, Confidence: {confidence_text}")
        
        # Create the result message with appropriate styling
        result_message = ""
        if label == "Fake":
            result_message = f"<div style='background-color: #ffebee; border-left: 5px solid #f44336; padding: 15px; margin: 10px 0; border-radius: 4px;'><span style='color: #d32f2f; font-weight: bold; font-size: 18px;'>⚠️ WARNING: AI-GENERATED AUDIO DETECTED</span><br><span style='color: #d32f2f;'>This audio file appears to be artificially generated with {confidence_text} confidence.</span></div>"
        else:
            result_message = f"<div style='background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 4px;'><span style='color: #2e7d32; font-weight: bold; font-size: 18px;'>✓ AUTHENTIC AUDIO</span><br><span style='color: #2e7d32;'>This audio file appears to be authentic with {confidence_text} confidence.</span></div>"
                
        return label, confidence_text, result_message
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {e}")
        logger.error(traceback.format_exc())
        error_msg = f"An error occurred: {str(e)}"
        return "Error", "0%", f"<div style='color: red;'>{error_msg}</div>"

demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Text(label="Confidence"),
        gr.HTML(label="Analysis Result")
    ],
    title="DeepFake Audio Detection",
    description="Upload an audio file to check if it's real or AI-generated (fake).",
    theme="dark",
    examples=[
        ["uploads/audio.wav"]
    ] if os.path.exists(os.path.join(UPLOAD_FOLDER, "audio.wav")) else None,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()