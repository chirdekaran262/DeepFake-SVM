import gradio as gr
import os
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import logging
import traceback
import datetime

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

def get_audio_info(audio_path):
    """Get duration and other information about the audio file"""
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sr)
        return {
            "duration": f"{duration:.2f} seconds",
            "sample_rate": f"{sr} Hz",
            "file_size": f"{os.path.getsize(audio_path) / 1024:.2f} KB"
        }
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return {
            "duration": "Unknown",
            "sample_rate": "Unknown",
            "file_size": "Unknown"
        }

def analyze_audio(audio_file):
    try:
        if audio_file is None:
            error_msg = "No file uploaded"
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
        if isinstance(audio_file, str):
            if not os.path.exists(audio_file):
                error_msg = f"File not found: {audio_file}"
                return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
            
            file_ext = os.path.splitext(audio_file)[1].lower().replace('.', '')
            if file_ext not in ALLOWED_EXTENSIONS:
                error_msg = f"Invalid file format: {file_ext}. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
                return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
            temp_path = audio_file
        else:
            try:
                if hasattr(audio_file, 'name'):
                    if not allowed_file(audio_file.name):
                        error_msg = f"Invalid file format. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
                        return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
                    
                    # Save uploaded file temporarily
                    temp_path = os.path.join(UPLOAD_FOLDER, os.path.basename(audio_file.name))
                    with open(temp_path, "wb") as f:
                        f.write(audio_file)
                else:
                    if isinstance(audio_file, tuple) and len(audio_file) >= 1:
                        temp_path = audio_file[0]
                    else:
                        error_msg = "Invalid audio format"
                        return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
        if model is None or scaler is None:
            error_msg = "Server configuration error. Models not loaded properly."
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
        # Get audio file information
        audio_info = get_audio_info(temp_path)
        file_name = os.path.basename(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            error_msg = "Could not extract features from the audio"
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Prediction: {label}, Confidence: {confidence_text}")
        
        # Status indicator HTML
        status_indicator = ""
        if label == "Real":
            status_indicator = """
            <div class="status-indicator real">
                <div class="indicator-dot"></div>
                <span>Real</span>
            </div>
            """
        else:
            status_indicator = """
            <div class="status-indicator fake">
                <div class="indicator-dot"></div>
                <span>Fake</span>
            </div>
            """
        
        # Create the result message with appropriate styling
        result_message = ""
        if label == "Fake":
            result_message = f"""
            <div class="result-card fake">
                <div class="result-header">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">AI-GENERATED AUDIO DETECTED</div>
                </div>
                <div class="result-body">
                    <p>This audio file appears to be artificially generated with <span class="highlight">{confidence_text}</span> confidence.</p>
                    <div class="details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Analyzed</div>
                            <div class="detail-value">{timestamp}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">File</div>
                            <div class="detail-value">{file_name}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Duration</div>
                            <div class="detail-value">{audio_info['duration']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Sample Rate</div>
                            <div class="detail-value">{audio_info['sample_rate']}</div>
                        </div>
                    </div>
                </div>
            </div>
            """
        else:
            result_message = f"""
            <div class="result-card real">
                <div class="result-header">
                    <div class="result-icon">✓</div>
                    <div class="result-title">AUTHENTIC AUDIO</div>
                </div>
                <div class="result-body">
                    <p>This audio file appears to be authentic with <span class="highlight">{confidence_text}</span> confidence.</p>
                    <div class="details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Analyzed</div>
                            <div class="detail-value">{timestamp}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">File</div>
                            <div class="detail-value">{file_name}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Duration</div>
                            <div class="detail-value">{audio_info['duration']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Sample Rate</div>
                            <div class="detail-value">{audio_info['sample_rate']}</div>
                        </div>
                    </div>
                </div>
            </div>
            """
                
        return label, confidence_text, result_message, status_indicator, audio_info
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {e}")
        logger.error(traceback.format_exc())
        error_msg = f"An error occurred: {str(e)}"
        return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}

modern_css = """
html, body, .gradio-container {
    min-height: 100vh;
    height: 100%;
    margin: 0;
    padding: 0;
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    color: #1a202c;
}
.gradio-container {
    min-height: 100vh !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.responsive-row {
    display: flex;
    flex-direction: row;
    gap: 2rem;
    width: 100%;
    max-width: 1100px;
    justify-content: center;
    align-items: stretch;
    height: 100vh;
    min-height: 600px;
    margin: auto; /* Center the row vertically and horizontally */
}
.centered-col, .result-col {
    flex: 1 1 0;
    min-width: 340px;
    max-width: 520px;
    min-height: 520px;
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 6px 32px rgba(99,102,241,0.10);
    padding: 2.5rem 2rem 2rem 2rem;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    /* Remove margin: auto 0; */
    box-sizing: border-box;
}
h1 {
    text-align: center;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #3730a3;
    letter-spacing: -1px;
}
h2 {
    font-size: 1.18rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
    color: #6366f1;
    text-align: center;
}
.upload-label {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #3730a3;
}
.gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}
.result-card {
    margin: 2rem 0 0 0;
    border-radius: 14px;
    background: #f1f5f9;
    border: 1.5px solid #c7d2fe;
    box-shadow: 0 2px 8px rgba(99,102,241,0.08);
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: box-shadow 0.2s;
    font-color: #1e293b;
}
.result-card.real {
    border-color: #10b981;
    background: #00ff00;
}
.result-card.fake {
    border-color: #ef4444;
    background: #ff0000;
}
.result-title {
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-confidence {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.5rem 0 1rem 0;
    color: #6366f1;
}
.result-card.real .result-title, .result-card.real .result-confidence {
    color: #059669;
}
.result-card.fake .result-title, .result-card.fake .result-confidence {
    color: #dc2626;
}
.result-body {
    color: #334155;
}
.result-card.real .result-body {
    color: #047857;
}
.result-card.fake .result-body {
    color: #b91c1c;
}
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 1rem 0;
     
 }
.status-indicator .indicator-dot {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    font-color: #fff;
     
}
.status-indicator.real .indicator-dot {
    background: #10b981;
    color: green;
    
}
.status-indicator.fake .indicator-dot {
    background: #ef4444;
    color: green;
    
}
.audio-info-table {
    width: 100%;
    margin: 1.5rem 0 0 0;
    border-collapse: collapse;
    font-size: 1rem;
}
.audio-info-table td {
    padding: 0.4rem 0.6rem;
    color: #475569;
}
.audio-info-table td:first-child {
    font-weight: 600;
    color: #1e293b;
}
.gradio-container .gr-button-primary {
    background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.8rem 1.7rem !important;
    font-size: 1.12rem !important;
    margin: 1.2rem 0 0.5rem 0 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.10);
    border: none !important;
}
.gradio-container .gr-label {
    font-weight: 600;
    color: #6366f1;
}
.gradio-container .gr-text-input input {
    border-radius: 10px !important;
}
.error-message {
    background: #fee2e2;
    color: #dc2626;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ef4444;
    margin: 1rem 0;
    text-align: center;
    font-weight: 600;
}
.gradio-container .gr-audio {
    width: 100% !important;
    min-height: 120px !important;
    max-height: 180px !important;
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    border: 2px dashed #c7d2fe !important;
    margin-bottom: 1.2rem !important;
    box-sizing: border-box !important;
}
@media (max-width: 1100px) {
    .responsive-row {
        gap: 1rem;
        height: unset;
        min-height: unset;
        margin: 0;
    }
}
@media (max-width: 900px) {
    .responsive-row {
        flex-direction: column;
        gap: 0;
        align-items: stretch;
        height: unset;
        min-height: unset;
        margin: 0;
    }
    .centered-col, .result-col {
        max-width: 98vw;
        margin: 1.5rem auto 0 auto;
        padding: 1.5rem 0.7rem;
        min-height: unset;
    }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.05rem; }
}
@media (max-width: 600px) {
    .responsive-row {
        flex-direction: column;
        gap: 0;
        align-items: stretch;
        height: unset;
        min-height: unset;
        margin: 0;
    }
    .centered-col, .result-col {
        max-width: 100vw;
        padding: 1rem 0.2rem;
        border-radius: 0;
        margin: 0;
        min-height: unset;
        box-shadow: none;
    }
    h1 { font-size: 1.1rem; }
    h2 { font-size: 0.97rem; }
}
"""

# Redesigned Gradio UI with responsive row/column layout
with gr.Blocks(css=modern_css, theme=gr.themes.Base()) as demo:
    gr.HTML("<h1>DeepFake Audio Detector</h1>")
    gr.HTML("<h2>Upload an audio file to check if it's genuine or AI-generated</h2>")
    with gr.Row(elem_classes="responsive-row"):
        with gr.Column(elem_classes="centered-col"):
            gr.HTML('<div class="upload-label">Audio File (WAV, MP3, OGG, FLAC):</div>')
            audio_input = gr.Audio(type="filepath", label="", elem_id="audio_input")
            analyze_btn = gr.Button("Analyze Audio", elem_id="analyze_btn", variant="primary")
            with gr.Accordion("Details", open=False):
                label_output = gr.Label(label="Classification")
                confidence_output = gr.Text(label="Confidence Score")
                audio_info_output = gr.JSON(label="Audio Info")
        with gr.Column(elem_classes="result-col"):
            result_output = gr.HTML(label="", elem_id="result_card")
            status_indicator_output = gr.HTML(label="", elem_id="status_indicator")
    # Footer
    gr.HTML("""
    <div style="text-align:center; color:#64748b; margin-top:2rem; font-size:0.97rem;">
        © 2025 DeepFake Audio Detector. For educational use only.
    </div>
    """)
    # Event handlers
    analyze_btn.click(
        fn=analyze_audio,
        inputs=audio_input,
        outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output]
    )
    audio_input.change(
        fn=analyze_audio,
        inputs=audio_input,
        outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output]
    )
    if os.path.exists(os.path.join(UPLOAD_FOLDER, "audio.wav")):
        gr.Examples(
            examples=[[os.path.join(UPLOAD_FOLDER, "audio.wav")]],
            inputs=audio_input,
            outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output],
            fn=analyze_audio,
            cache_examples=False
        )

if __name__ == "__main__":
    demo.launch()