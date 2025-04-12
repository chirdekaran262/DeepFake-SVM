from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if 'audio' not in request.files:
                logger.warning("No file part in the request")
                return jsonify({"success": False, "error": "No file part in the request"}), 400

            file = request.files['audio']
            if file.filename == '':
                logger.warning("No file selected")
                return jsonify({"success": False, "error": "No file selected"}), 400

            if not allowed_file(file.filename):
                logger.warning(f"Invalid file format: {file.filename}")
                return jsonify({"success": False, "error": f"Invalid file format. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

            if model is None or scaler is None:
                logger.error("Model or scaler not loaded properly")
                return jsonify({"success": False, "error": "Server configuration error. Please contact the administrator."}), 500

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved to {filepath}")

            features = extract_features(filepath)
            if features is None:
                return jsonify({"success": False, "error": "Could not extract features from the audio."}), 500

            try:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                
                # Check if model has predict_proba (for SVC with probability=True)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = probabilities[prediction] * 100
                    logger.info(f"Using predict_proba for confidence: {confidence}")
                else:
                    # Fallback to decision_function for models without probability calibration
                    confidence_score = np.max(np.abs(model.decision_function(features_scaled)))
                    # Scale confidence to a percentage (0-100)
                    confidence = min(max(confidence_score * 50, 50), 99)
                    logger.info(f"Using decision_function for confidence: {confidence}")

                label = "Real" if prediction == 0 else "Fake"
                logger.info(f"Prediction: {label}, Confidence: {confidence}")

                # Additional details for the frontend
                details = {
                    "audio_length_seconds": round(librosa.get_duration(path=filepath), 2),
                    "sample_rate": librosa.get_samplerate(filepath),
                    "model_type": "Enhanced (MFCC + Spectral Contrast + Chroma)" if "enhanced" in str(model) else "Basic (MFCC only)"
                }

                return jsonify({
                    "success": True,
                    "prediction": label,
                    "confidence": round(confidence, 2),
                    "details": details
                })
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                logger.error(traceback.format_exc())
                return jsonify({"success": False, "error": "Error analyzing the audio file"}), 500
            
        except Exception as e:
            logger.error(f"General error in POST handling: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"success": False, "error": "An unexpected error occurred"}), 500

    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    status = "OK" if model is not None and scaler is not None else "ERROR"
    model_type = "enhanced" if "enhanced" in str(model) else "basic"
    return jsonify({
        "status": status, 
        "model_type": model_type,
        "features": "MFCC + Spectral Contrast + Chroma" if model_type == "enhanced" else "MFCC only"
    })

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for programmatic access"""
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No file part in the request"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": f"Invalid file format"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        features = extract_features(filepath)
        if features is None:
            return jsonify({"success": False, "error": "Could not extract features from the audio"}), 500

        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        
        # Get confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction] * 100
        else:
            confidence_score = np.max(np.abs(model.decision_function(features_scaled)))
            confidence = min(max(confidence_score * 50, 50), 99)

        result = {
            "success": True,
            "prediction": "Real" if prediction == 0 else "Fake",
            "confidence": round(confidence, 2),
            "file_analyzed": filename
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "error": "The audio file is too large. Please upload a file smaller than 16MB"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({"success": False, "error": "Server error occurred. Please try again later."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
