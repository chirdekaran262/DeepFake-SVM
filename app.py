# import gradio as gr
# import os
# import librosa
# import numpy as np
# import joblib
# from werkzeug.utils import secure_filename
# import logging
# import traceback
# import datetime

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
# UPLOAD_FOLDER = 'uploads'

# # Ensure upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load model and scaler
# try:
#     model = joblib.load("enhanced_svm_model.pkl")
#     scaler = joblib.load("enhanced_scaler.pkl")
#     logger.info("Enhanced model and scaler loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading enhanced model files: {e}")
#     try:
#         # Fallback to original model
#         model = joblib.load("svm_model.pkl")
#         scaler = joblib.load("scaler.pkl")
#         logger.info("Fallback to original model successful")
#     except Exception as e2:
#         logger.error(f"Error loading fallback model files: {e2}")
#         model = None
#         scaler = None

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
#     try:
#         audio_data, sr = librosa.load(audio_path, sr=None)
        
#         # Extract MFCC features
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         mfccs_mean = np.mean(mfccs.T, axis=0)
        
#         # Extract Spectral Contrast
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
#         spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
#         # Extract Chroma Features
#         chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
#         chroma_mean = np.mean(chroma.T, axis=0)
        
#         # Combine all features
#         combined_features = np.concatenate((mfccs_mean, spectral_contrast_mean, chroma_mean))
        
#         return combined_features
#     except Exception as e:
#         logger.error(f"Error extracting features: {e}")
#         logger.error(traceback.format_exc())
#         return None

# def get_audio_info(audio_path):
#     """Get duration and other information about the audio file"""
#     try:
#         audio_data, sr = librosa.load(audio_path, sr=None)
#         duration = librosa.get_duration(y=audio_data, sr=sr)
#         return {
#             "duration": f"{duration:.2f} seconds",
#             "sample_rate": f"{sr} Hz",
#             "file_size": f"{os.path.getsize(audio_path) / 1024:.2f} KB"
#         }
#     except Exception as e:
#         logger.error(f"Error getting audio info: {e}")
#         return {
#             "duration": "Unknown",
#             "sample_rate": "Unknown",
#             "file_size": "Unknown"
#         }

# def analyze_audio(audio_file):
#     try:
#         if audio_file is None:
#             error_msg = "No file uploaded"
#             return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
#         if isinstance(audio_file, str):
#             if not os.path.exists(audio_file):
#                 error_msg = f"File not found: {audio_file}"
#                 return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
            
#             file_ext = os.path.splitext(audio_file)[1].lower().replace('.', '')
#             if file_ext not in ALLOWED_EXTENSIONS:
#                 error_msg = f"Invalid file format: {file_ext}. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
#                 return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
#             temp_path = audio_file
#         else:
#             try:
#                 if hasattr(audio_file, 'name'):
#                     if not allowed_file(audio_file.name):
#                         error_msg = f"Invalid file format. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
#                         return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
                    
#                     # Save uploaded file temporarily
#                     temp_path = os.path.join(UPLOAD_FOLDER, os.path.basename(audio_file.name))
#                     with open(temp_path, "wb") as f:
#                         f.write(audio_file)
#                 else:
#                     if isinstance(audio_file, tuple) and len(audio_file) >= 1:
#                         temp_path = audio_file[0]
#                     else:
#                         error_msg = "Invalid audio format"
#                         return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
#             except Exception as e:
#                 error_msg = f"Error processing file: {str(e)}"
#                 return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
#         if model is None or scaler is None:
#             error_msg = "Server configuration error. Models not loaded properly."
#             return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
#         # Get audio file information
#         audio_info = get_audio_info(temp_path)
#         file_name = os.path.basename(temp_path)
        
#         # Extract features
#         features = extract_features(temp_path)
#         if features is None:
#             error_msg = "Could not extract features from the audio"
#             return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}
        
#         # Make prediction
#         features_scaled = scaler.transform(features.reshape(1, -1))
#         prediction = model.predict(features_scaled)[0]
        
#         # Calculate confidence
#         if hasattr(model, 'predict_proba'):
#             probabilities = model.predict_proba(features_scaled)[0]
#             confidence = probabilities[prediction] * 100
#             logger.info(f"Using predict_proba for confidence: {confidence}")
#         else:
#             confidence_score = np.max(np.abs(model.decision_function(features_scaled)))
#             confidence = min(max(confidence_score * 50, 50), 99)
#             logger.info(f"Using decision_function for confidence: {confidence}")
        
#         label = "Real" if prediction == 0 else "Fake"
#         confidence_text = f"{confidence:.2f}%"
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         logger.info(f"Prediction: {label}, Confidence: {confidence_text}")
        
#         # Status indicator HTML
#         status_indicator = ""
#         if label == "Real":
#             status_indicator = """
#             <div class="status-indicator real">
#                 <div class="indicator-dot"></div>
#                 <span>Real</span>
#             </div>
#             """
#         else:
#             status_indicator = """
#             <div class="status-indicator fake">
#                 <div class="indicator-dot"></div>
#                 <span>Fake</span>
#             </div>
#             """
        
#         # Create the result message with appropriate styling
#         result_message = ""
#         if label == "Fake":
#             result_message = f"""
#             <div class="result-card fake">
#                 <div class="result-header">
#                     <div class="result-icon">⚠️</div>
#                     <div class="result-title">AI-GENERATED AUDIO DETECTED</div>
#                 </div>
#                 <div class="result-body">
#                     <p>This audio file appears to be artificially generated with <span class="highlight">{confidence_text}</span> confidence.</p>
#                     <div class="details-grid">
#                         <div class="detail-item">
#                             <div class="detail-label">Analyzed</div>
#                             <div class="detail-value">{timestamp}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">File</div>
#                             <div class="detail-value">{file_name}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">Duration</div>
#                             <div class="detail-value">{audio_info['duration']}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">Sample Rate</div>
#                             <div class="detail-value">{audio_info['sample_rate']}</div>
#                         </div>
#                     </div>
#                 </div>
#             </div>
#             """
#         else:
#             result_message = f"""
#             <div class="result-card real">
#                 <div class="result-header">
#                     <div class="result-icon">✓</div>
#                     <div class="result-title">AUTHENTIC AUDIO</div>
#                 </div>
#                 <div class="result-body">
#                     <p>This audio file appears to be authentic with <span class="highlight">{confidence_text}</span> confidence.</p>
#                     <div class="details-grid">
#                         <div class="detail-item">
#                             <div class="detail-label">Analyzed</div>
#                             <div class="detail-value">{timestamp}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">File</div>
#                             <div class="detail-value">{file_name}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">Duration</div>
#                             <div class="detail-value">{audio_info['duration']}</div>
#                         </div>
#                         <div class="detail-item">
#                             <div class="detail-label">Sample Rate</div>
#                             <div class="detail-value">{audio_info['sample_rate']}</div>
#                         </div>
#                     </div>
#                 </div>
#             </div>
#             """
                
#         return label, confidence_text, result_message, status_indicator, audio_info
        
#     except Exception as e:
#         logger.error(f"Error in analyze_audio: {e}")
#         logger.error(traceback.format_exc())
#         error_msg = f"An error occurred: {str(e)}"
#         return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", {}

# modern_css = """
# html, body, .gradio-container {
#     min-height: 100vh;
#     height: 100%;
#     margin: 0;
#     padding: 0;
#     font-family: 'Inter', sans-serif;
#     background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
#     color: #1a202c;
# }
# .gradio-container {
#     min-height: 100vh !important;
#     display: flex;
#     flex-direction: column;
#     justify-content: center;
#     align-items: center;
# }
# .responsive-row {
#     display: flex;
#     flex-direction: row;
#     gap: 2rem;
#     width: 100%;
#     max-width: 1100px;
#     justify-content: center;
#     align-items: stretch;
#     height: 100vh;
#     min-height: 600px;
#     margin: auto; /* Center the row vertically and horizontally */
# }
# .centered-col, .result-col {
#     flex: 1 1 0;
#     min-width: 340px;
#     max-width: 520px;
#     min-height: 520px;
#     max-height: 620px;
#     background: #fff;
#     border-radius: 18px;
#     box-shadow: 0 6px 32px rgba(99,102,241,0.10);
#     padding: 2.5rem 2rem 2rem 2rem;
#     display: flex;
#     flex-direction: column;
#     align-items: stretch;
#     /* Remove margin: auto 0; */
#     box-sizing: border-box;
# }
# h1 {
#     text-align: center;
#     font-size: 2.3rem;
#     font-weight: 800;
#     margin-bottom: 0.5rem;
#     color: #3730a3;
#     letter-spacing: -1px;
# }
# h2 {
#     font-size: 1.18rem;
#     font-weight: 500;
#     margin-bottom: 1.5rem;
#     color: #6366f1;
#     text-align: center;
# }
# .upload-label {
#     font-size: 1.05rem;
#     font-weight: 600;
#     margin-bottom: 0.5rem;
#     color: #3730a3;
# }
# .gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
#     background: none !important;
#     border: none !important;
#     box-shadow: none !important;
# }
# .result-card {
#     margin: 2rem 0 0 0;
#     border-radius: 14px;
#     background: #f1f5f9;
#     border: 1.5px solid #c7d2fe;
#     box-shadow: 0 2px 8px rgba(99,102,241,0.08);
#     padding: 1.5rem 1.25rem;
#     text-align: center;
#     transition: box-shadow 0.2s;
#     font-color: #1e293b;
# }
# .result-card.real {
#     border-color: #10b981;
#     background: #00ff00;
# }
# .result-card.fake {
#     border-color: #ef4444;
#     background: #ff0000;
# }
# .result-title {
#     font-size: 1.18rem;
#     font-weight: 700;
#     margin-bottom: 0.5rem;
# }
# .result-confidence {
#     font-size: 1.5rem;
#     font-weight: 700;
#     margin: 0.5rem 0 1rem 0;
#     color: #6366f1;
# }
# .result-card.real .result-title, .result-card.real .result-confidence {
#     color: #059669;
# }
# .result-card.fake .result-title, .result-card.fake .result-confidence {
#     color: #dc2626;
# }
# .result-body {
#     color: #334155;
# }
# .result-card.real .result-body {
#     color: #047857;
# }
# .result-card.fake .result-body {
#     color: #b91c1c;
# }
# .status-indicator {
#     display: inline-flex;
#     align-items: center;
#     gap: 0.5rem;
#     font-size: 1.05rem;
#     font-weight: 700;
#     margin: 1rem 0;
     
#  }
# .status-indicator .indicator-dot {
#     width: 13px;
#     height: 13px;
#     border-radius: 50%;
#     font-color: #fff;
     
# }
# .status-indicator.real .indicator-dot {
#     background: #10b981;
#     color: green;
    
# }
# .status-indicator.fake .indicator-dot {
#     background: #ef4444;
#     color: green;
    
# }
# .audio-info-table {
#     width: 100%;
#     margin: 1.5rem 0 0 0;
#     border-collapse: collapse;
#     font-size: 1rem;
# }
# .audio-info-table td {
#     padding: 0.4rem 0.6rem;
#     color: #475569;
# }
# .audio-info-table td:first-child {
#     font-weight: 600;
#     color: #1e293b;
# }
# .gradio-container .gr-button-primary {
#     background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%) !important;
#     color: #fff !important;
#     border-radius: 10px !important;
#     font-weight: 700 !important;
#     padding: 0.8rem 1.7rem !important;
#     font-size: 1.12rem !important;
#     margin: 1.2rem 0 0.5rem 0 !important;
#     box-shadow: 0 2px 8px rgba(99,102,241,0.10);
#     border: none !important;
# }
# .gradio-container .gr-label {
#     font-weight: 600;
#     color: #6366f1;
# }
# .gradio-container .gr-text-input input {
#     border-radius: 10px !important;
# }
# .error-message {
#     background: #fee2e2;
#     color: #dc2626;
#     padding: 1rem;
#     border-radius: 10px;
#     border-left: 5px solid #ef4444;
#     margin: 1rem 0;
#     text-align: center;
#     font-weight: 600;
# }
# .gradio-container .gr-audio {
#     width: 100% !important;
#     min-height: 120px !important;
#     max-height: 180px !important;
#     background: #f1f5f9 !important;
#     border-radius: 12px !important;
#     border: 2px dashed #c7d2fe !important;
#     margin-bottom: 1.2rem !important;
#     box-sizing: border-box !important;
# }
# @media (max-width: 1100px) {
#     .responsive-row {
#         gap: 1rem;
#         height: unset;
#         min-height: unset;
#         margin: 0;
#     }
# }
# @media (max-width: 900px) {
#     .responsive-row {
#         flex-direction: column;
#         gap: 0;
#         align-items: stretch;
#         height: unset;
#         min-height: unset;
#         margin: 0;
#     }
#     .centered-col, .result-col {
#         max-width: 98vw;
#         margin: 1.5rem auto 0 auto;
#         padding: 1.5rem 0.7rem;
#         min-height: unset;
#     }
#     h1 { font-size: 1.5rem; }
#     h2 { font-size: 1.05rem; }
# }
# @media (max-width: 600px) {
#     // ...existing code...
# .responsive-row {
#     display: flex;
#     flex-direction: row;
#     gap: 2rem;
#     width: 100%;
#     max-width: 1100px;
#     justify-content: center;
#     align-items: stretch;
#     height: 65vh; /* Reduced from 80vh */
#     min-height: 400px; /* Reduced from 500px */
#     margin: auto;
# }


# .centered-col, .result-col {
#     flex: 1 1 0;
#     min-width: 340px;
#     max-width: 520px;
#     height: 350px; /* Reduced from 450px */
#     background: #fff;
#     border-radius: 18px;
#     box-shadow: 0 6px 32px rgba(99,102,241,0.10);
#     padding: 1.5rem 1.5rem; /* Reduced padding */
#     display: flex;
#     flex-direction: column;
#     align-items: stretch;
#     box-sizing: border-box;
#     overflow-y: auto;
# }

# @media (max-width: 900px) {
#     .centered-col, .result-col {
#         height: auto;
#         min-height: 300px; /* Reduced from 400px */
#         margin: 0.8rem auto;
#         padding: 1.2rem 0.7rem;
#     }
# }
# // ...existing code...
#     h1 { font-size: 1.1rem; }
#     h2 { font-size: 0.97rem; }
# }
# // Add this CSS to your modern_css string
# .footer {
#     width: 100%;
#     background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
#     padding: 2rem 1rem;
#     margin-top: 3rem;
#     box-shadow: 0 -4px 16px rgba(99,102,241,0.1);
#     position: relative;  /* Add this */
#     z-index: 10;        /* Add this */
# }

# .footer-content {
#     max-width: 1200px;
#     margin: 0 auto;
#     display: grid;
#     grid-template-columns: repeat(3, 1fr);
#     gap: 2rem;
#     color: #fff;
#     background: inherit;  /* Add this */
# }

# .footer-section {
#     padding: 0 1rem;
#     background: transparent;  /* Add this */
# }

# /* Remove the background-color: coral; from the footer class as it conflicts with the gradient */

# .footer-bottom {
#     text-align: center;
#     padding-top: 2rem;
#     margin-top: 2rem;
#     border-top: 1px solid rgba(255,255,255,0.2);  /* Made more visible */
#     color: #fff;           /* Changed to pure white */
#     font-size: 0.9rem;
#     background: inherit;   /* Add this */
# }

# /* Add this new style for stronger text contrast */
# .footer-section h3,
# .footer-section p,
# .footer-section a {
#     color: #fff;
#     text-shadow: 0 1px 2px rgba(0,0,0,0.1);
# }

# .social-links a {
#     color: #fff;
#     font-size: 1.5rem;    /* Made slightly larger */
#     text-shadow: 0 1px 2px rgba(0,0,0,0.1);
#     transition: transform 0.2s ease;
# }

# .social-links a:hover {
#     transform: translateY(-2px);
# }
# """

# # Redesigned Gradio UI with responsive row/column layout
# with gr.Blocks(css=modern_css, theme=gr.themes.Base()) as demo:
#     gr.HTML("<h1>DeepFake Audio Detector</h1>")
#     gr.HTML("<h2>Upload an audio file to check if it's genuine or AI-generated</h2>")
#     with gr.Row(elem_classes="responsive-row"):
#         with gr.Column(elem_classes="centered-col"):
#             gr.HTML('<div class="upload-label">Audio File (WAV, MP3, OGG, FLAC):</div>')
#             audio_input = gr.Audio(type="filepath", label="", elem_id="audio_input")
#             analyze_btn = gr.Button("Analyze Audio", elem_id="analyze_btn", variant="primary")
#             with gr.Accordion("Details", open=False):
#                 label_output = gr.Label(label="Classification")
#                 confidence_output = gr.Text(label="Confidence Score")
#                 audio_info_output = gr.JSON(label="Audio Info")
#         with gr.Column(elem_classes="result-col"):
#             result_output = gr.HTML(label="", elem_id="result_card")
#             status_indicator_output = gr.HTML(label="", elem_id="status_indicator")
#     # Footer
#     # Replace the existing footer HTML with this
#     gr.HTML("""
#         <div class="footer">
#             <div class="footer-content">
#                 <div class="footer-section">
#                     <h3>About Us</h3>
#                     <p>DeepFake Audio Detector is an advanced AI-powered tool designed to identify artificially generated audio content with high accuracy.</p>
#                 </div>
#                 <div class="footer-section">
#                     <h3>Quick Links</h3>
#                     <p><a href="#">Documentation</a></p>
#                     <p><a href="#">Privacy Policy</a></p>
#                     <p><a href="#">Terms of Service</a></p>
#                 </div>
#                 <div class="footer-section">
#                     <h3>Contact</h3>
#                     <p>Email: info@deepfakedetector.ai</p>
#                     <p>Support: support@deepfakedetector.ai</p>
#                     <div class="social-links">
#                         <a href="#" title="Twitter">🐦</a>
#                         <a href="#" title="GitHub">📚</a>
#                         <a href="#" title="LinkedIn">💼</a>
#                     </div>
#                 </div>
#             </div>
#             <div class="footer-bottom">
#                 <p>© 2025 DeepFake Audio Detector. All rights reserved.</p>
#             </div>
#         </div>
        
#     """)
#     # Event handlers
#     analyze_btn.click(
#         fn=analyze_audio,
#         inputs=audio_input,
#         outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output]
#     )
#     audio_input.change(
#         fn=analyze_audio,
#         inputs=audio_input,
#         outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output]
#     )
#     if os.path.exists(os.path.join(UPLOAD_FOLDER, "audio.wav")):
#         gr.Examples(
#             examples=[[os.path.join(UPLOAD_FOLDER, "audio.wav")]],
#             inputs=audio_input,
#             outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output],
#             fn=analyze_audio,
#             cache_examples=False
#         )

# if __name__ == "__main__":
#     demo.launch()

import gradio as gr
import os
import librosa
import numpy as np
import joblib
# from werkzeug.utils import secure_filename # Not needed
import logging
import traceback
import datetime
import tempfile
import soundfile as sf # Use soundfile for info and dummy file creation

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# --- Model Loading ---
MODEL_LOAD_STATUS = "Models loaded successfully."
try:
    model = joblib.load("enhanced_svm_model.pkl")
    scaler = joblib.load("enhanced_scaler.pkl")
    logger.info("Enhanced model and scaler loaded successfully")
except Exception as e:
    logger.warning(f"Could not load enhanced model/scaler: {e}. Trying fallback...")
    MODEL_LOAD_STATUS = f"Warning: Could not load enhanced model ({e}). Trying fallback."
    try:
        model = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("Fallback to original model successful")
        MODEL_LOAD_STATUS = "Using fallback model."
    except Exception as e2:
        logger.error(f"FATAL: Error loading fallback model files: {e2}")
        model = None
        scaler = None
        MODEL_LOAD_STATUS = f"ERROR: Failed to load any models ({e2}). App may not function."

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None, duration=30) # Limit duration

        if len(audio_data) < hop_length:
             logger.warning(f"Audio file {audio_path} is too short for feature extraction.")
             return None

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma.T, axis=0)
        combined_features = np.concatenate((mfccs_mean, spectral_contrast_mean, chroma_mean))
        return combined_features
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def get_audio_info(audio_path):
    """Get duration and other information about the audio file"""
    try:
        info = sf.info(audio_path)
        duration = info.duration
        sr = info.samplerate
        channels = info.channels
        file_size = os.path.getsize(audio_path)
        return {
            "Duration": f"{duration:.2f} sec",
            "Sample Rate": f"{sr} Hz",
            "Channels": channels,
            "File Size": f"{file_size / 1024:.1f} KB",
            "File Name": os.path.basename(audio_path)
        }
    except Exception as e:
        logger.error(f"Error getting audio info for {audio_path}: {e}")
        # Basic fallback
        try:
            file_size = os.path.getsize(audio_path)
            return {
                "Duration": "N/A", "Sample Rate": "N/A", "Channels": "N/A",
                "File Size": f"{file_size / 1024:.1f} KB",
                "File Name": os.path.basename(audio_path)
            }
        except Exception: # If even size fails
             return {
                "Duration": "Error", "Sample Rate": "Error", "Channels": "Error",
                "File Size": "Error", "File Name": os.path.basename(audio_path)
            }

# --- Main Analysis Function ---
def analyze_audio(audio_filepath):
    initial_result_html = "<div class='result-placeholder'>Upload an audio file and click Analyze</div>"
    initial_status_html = ""
    initial_audio_info = {}
    initial_label = "N/A"
    initial_confidence = "0%"

    if audio_filepath is None:
        # Return defaults gracefully when no file is uploaded yet or cleared
        return initial_label, initial_confidence, initial_result_html, initial_status_html, initial_audio_info

    try:
        file_ext = os.path.splitext(audio_filepath)[1].lower().replace('.', '')
        if file_ext not in ALLOWED_EXTENSIONS:
            error_msg = f"Invalid file format: .{file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            logger.warning(error_msg)
            # Return specific error message but keep other outputs clean
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", "", get_audio_info(audio_filepath)


        if model is None or scaler is None:
            error_msg = "Server Configuration Error: Models not loaded. Cannot analyze."
            logger.error(error_msg)
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", initial_status_html, get_audio_info(audio_filepath)

        audio_info = get_audio_info(audio_filepath)
        file_name = audio_info.get("File Name", os.path.basename(audio_filepath))

        logger.info(f"Extracting features for: {file_name}")
        features = extract_features(audio_filepath)
        if features is None:
            error_msg = f"Could not extract features: {file_name}. File might be corrupted, too short, or silent."
            logger.error(error_msg)
            return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", initial_status_html, audio_info

        logger.info("Scaling features and making prediction...")
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0] # 0 for Real, 1 for Fake

        confidence = 50.0 # Default confidence
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = probabilities[prediction] * 100
                logger.info(f"Confidence from predict_proba: {confidence:.2f}%")
            elif hasattr(model, 'decision_function'):
                decision_score = model.decision_function(features_scaled)[0]
                 # Slightly adjusted scaling for potentially better spread
                confidence = np.clip(50 + abs(decision_score) * 20, 50, 99.5)
                logger.info(f"Confidence derived from decision_function (score {decision_score:.3f}): {confidence:.2f}%")
            else:
                 logger.warning("Model lacks confidence methods. Using default 50%.")
        except Exception as conf_e:
            logger.error(f"Error calculating confidence: {conf_e}")
            logger.warning("Using default 50% confidence due to error.")


        label = "Real" if prediction == 0 else "Fake"
        confidence_text = f"{confidence:.1f}%"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Prediction complete: Label={label}, Confidence={confidence_text}, File={file_name}")

        status_indicator_html = ""
        result_html = ""
        result_class = "real" if label == "Real" else "fake"
        result_icon = "✓" if label == "Real" else "⚠️"
        result_title = "LIKELY AUTHENTIC" if label == "Real" else "POTENTIALLY AI-GENERATED"
        status_text = "Authentic" if label == "Real" else "AI-Generated"
        result_message = "This audio file appears to be authentic." if label == "Real" else "This audio file shows characteristics of being AI-generated."

        status_indicator_html = f"""
        <div class="status-indicator {result_class}">
            <div class="indicator-dot"></div>
            <span>{status_text}</span>
        </div>
        """
        result_html = f"""
        <div class="result-card {result_class}">
            <div class="result-header">
                <div class="result-icon">{result_icon}</div>
                <div class="result-title">{result_title}</div>
            </div>
            <div class="result-body">
                <p>{result_message}</p>
                <p class="result-confidence">Confidence: <span class="highlight">{confidence_text}</span></p>
                <div class="details-summary">
                    Analyzed: {timestamp}
                </div>
            </div>
        </div>
        """

        return label, confidence_text, result_html, status_indicator_html, audio_info

    except librosa.LibrosaError as lib_e:
         logger.error(f"Librosa error processing {audio_filepath}: {lib_e}")
         error_msg = f"Audio Processing Error: Could not load/process the file. It might be corrupted or an unsupported format variant. ({lib_e})"
         return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", initial_status_html, get_audio_info(audio_filepath)
    except FileNotFoundError:
        logger.error(f"File not found error for: {audio_filepath}")
        error_msg = "Error: The uploaded file reference seems broken."
        return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", initial_status_html, initial_audio_info
    except Exception as e:
        logger.error(f"Unexpected error in analyze_audio for {audio_filepath}: {e}")
        logger.error(traceback.format_exc())
        error_msg = f"An unexpected server error occurred: {str(e)}"
        # Try to get info even if analysis fails later
        audio_info_on_error = {}
        if audio_filepath and os.path.exists(audio_filepath):
             audio_info_on_error = get_audio_info(audio_filepath)
        return "Error", "0%", f"<div class='error-message'>{error_msg}</div>", initial_status_html, audio_info_on_error


# --- Enhanced CSS ---
modern_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary-color: #6366f1; /* Indigo 500 */
    --primary-hover: #4f46e5; /* Indigo 600 */
    --primary-light: #e0e7ff; /* Indigo 100 */
    --real-color: #16a34a; /* Green 600 */
    --real-bg: #f0fdf4;    /* Green 50 */
    --real-border: #bbf7d0;/* Green 200 */
    --fake-color: #dc2626; /* Red 600 */
    --fake-bg: #fef2f2;    /* Red 50 */
    --fake-border: #fecaca;/* Red 200 */
    --text-dark: #1f2937;  /* Gray 800 */
    --text-medium: #4b5563;/* Gray 600 */
    --text-light: #6b7280; /* Gray 500 */
    --bg-main: #f9fafb;    /* Gray 50 - Slightly off-white for main bg */
    --bg-card: #ffffff;   /* White */
    --border-color: #e5e7eb; /* Gray 200 */
    --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    --card-shadow-lg: 0 10px 15px -3px rgba(99, 102, 241, 0.08), 0 4px 6px -2px rgba(99, 102, 241, 0.04);
    --card-radius: 12px;
    --input-radius: 8px;
}

html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    scroll-behavior: smooth;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-dark);
    display: flex;
    flex-direction: column;
    min-height: 100vh; /* Ensure footer stays down */
}

/* Remove default Gradio backgrounds/borders */
.gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Header Styling */
.app-header {
    background: var(--bg-card);
    padding: 0.8rem 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
    position: sticky; /* Keep header visible */
    top: 0;
    z-index: 100;
}
.logo {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary-hover);
}
.logo span {
    font-weight: 400;
    color: var(--text-medium);
}
.nav-links a {
    margin-left: 1.5rem;
    text-decoration: none;
    color: var(--text-medium);
    font-weight: 500;
    font-size: 0.95rem;
    transition: color 0.2s ease;
}
.nav-links a:hover {
    color: var(--primary-color);
}

/* Main Content Area - Centering the Row */
.gradio-container > .flex.flex-col { /* Target Gradio's main content wrapper */
    flex-grow: 1; /* Take available space */
    display: flex;
    flex-direction: column; /* Stack header, content, footer */
    /* align-items: center; /* Center children horizontally */
}

/* Wrapper specifically for the content row to center it */
.main-content-wrapper {
    flex-grow: 1; /* Takes space between header and footer */
    display: flex;
    justify-content: center; /* Center horizontally */
    align-items: center; /* Center vertically */
    padding: 2rem 1rem; /* Padding around the centered content */
    width: 100%;
    box-sizing: border-box;
}

/* Responsive Row for Input/Output columns */
.responsive-row {
    display: flex;
    flex-direction: row;
    gap: 2rem;
    width: 100%;
    max-width: 1000px; /* Max width of the two columns together */
    /* Removed align-items: stretch to allow natural height */
}

/* Columns Styling */
.input-col, .result-col {
    flex: 1 1 0px; /* Allow flex grow/shrink, base width 0 */
    min-width: 300px; /* Minimum width before wrapping */
    background: var(--bg-card);
    border-radius: var(--card-radius);
    box-shadow: var(--card-shadow-lg);
    padding: 1.75rem 1.5rem; /* Slightly reduced padding */
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
    transition: box-shadow 0.3s ease;
}
.input-col:hover, .result-col:hover {
    /* box-shadow: var(--card-shadow-lg); */ /* Keep consistent shadow or enhance slightly */
}

h1, h2 { /* Remove default h1/h2 as they are in header/intro */
   display: none;
}

/* Intro Text (Optional - Placed before the row) */
.intro-text {
    text-align: center;
    margin-bottom: 2rem; /* Space before the boxes */
    max-width: 650px; /* Limit width */
    margin-left: auto;
    margin-right: auto;
}
.intro-text h1 { /* Style the new intro h1 */
    display: block; /* Make it visible again */
    font-size: 2.0rem;
    font-weight: 800;
    color: var(--primary-hover);
    margin-bottom: 0.5rem;
}
.intro-text h2 { /* Style the new intro h2 */
    display: block; /* Make it visible again */
    font-size: 1.05rem;
    color: var(--text-medium);
    font-weight: 400;
    line-height: 1.6;
}


/* Input Column Specifics */
.upload-label {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--primary-hover);
}

.gradio-container .gr-audio {
    width: 100% !important;
    background: #f9fafb !important; /* Lighter gray bg */
    border-radius: var(--input-radius) !important;
    border: 2px dashed var(--primary-light) !important; /* Lighter dashed border */
    margin-bottom: 1rem !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}
.gradio-container .gr-audio:hover {
    border-color: var(--primary-color) !important;
    background-color: #f5f6fe !important; /* Slightly bluer on hover */
}

.gradio-container .gr-button-primary {
    background: var(--primary-color) !important;
    color: #fff !important;
    border-radius: var(--input-radius) !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important;
    margin: 1rem 0 0.5rem 0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    border: none !important;
    width: 100%;
    transition: background-color 0.2s ease, transform 0.1s ease;
}
.gradio-container .gr-button-primary:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px); /* Slight lift on hover */
}

/* Accordion styling */
.gradio-container .gr-accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--input-radius) !important;
    margin-top: 1.25rem;
    padding: 0 !important;
    box-shadow: none;
    overflow: hidden; /* Ensures contained border radius */
}
.gradio-container .gr-accordion > button { /* Accordion header */
    background: #f9fafb !important;
    padding: 0.6rem 1rem !important;
    border-bottom: 1px solid var(--border-color);
    font-weight: 500;
    font-size: 0.9rem;
    color: var(--text-medium);
    transition: background-color 0.2s ease;
}
.gradio-container .gr-accordion > button:hover {
     background: #f3f4f6 !important;
}
.gradio-container .gr-accordion > div { /* Accordion content */
     padding: 0.75rem 1rem 0.75rem 1rem !important;
     background: var(--bg-card) !important;
}
.gradio-container .gr-json {
    padding: 0.5rem 0.75rem !important;
    background-color: #f9fafb !important;
    border-radius: 6px !important;
    font-size: 0.8rem; /* Smaller font for details */
    border: 1px solid #f3f4f6; /* Softer border */
    color: var(--text-medium);
    white-space: pre-wrap; /* Allow wrapping */
    word-break: break-all;
}
/* Pre-format JSON keys for slight emphasis */
.gradio-container .gr-json pre span.hljs-attr {
    color: var(--primary-color) !important;
    font-weight: 500;
}


/* Result Column Specifics */
.result-col {
    justify-content: flex-start;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.0rem; /* Slightly smaller */
    font-weight: 600;
    margin-bottom: 1rem; /* Reduced space */
    padding: 0.4rem 0.75rem;
    border-radius: 6px;
    align-self: flex-start; /* Align to the left */
}
.status-indicator .indicator-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.status-indicator.real { color: var(--real-color); background-color: var(--real-bg); }
.status-indicator.real .indicator-dot { background-color: var(--real-color); }
.status-indicator.fake { color: var(--fake-color); background-color: var(--fake-bg); }
.status-indicator.fake .indicator-dot { background-color: var(--fake-color); }

/* Result Card Styling */
.result-placeholder {
    text-align: center;
    color: var(--text-light);
    padding: 3rem 1rem;
    font-size: 0.95rem;
    border: 2px dashed var(--border-color);
    border-radius: var(--input-radius);
    margin-top: 1rem; /* Add margin if status indicator is not present initially */
    flex-grow: 1; /* Allow placeholder to take space */
    display: flex;
    align-items: center;
    justify-content: center;
}

.result-card {
    border-radius: var(--input-radius);
    border: 1px solid;
    padding: 1.25rem 1.25rem; /* Uniform padding */
    text-align: center;
    transition: all 0.3s ease;
    width: 100%;
    box-sizing: border-box;
    margin-top: 0; /* Removed top margin, rely on status indicator margin */
}
.result-card.real { background-color: var(--real-bg); border-color: var(--real-border); color: var(--real-color); }
.result-card.fake { background-color: var(--fake-bg); border-color: var(--fake-border); color: var(--fake-color); }

.result-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
}
.result-icon { font-size: 1.5rem; line-height: 1; }
.result-title { font-size: 1.0rem; font-weight: 700; margin: 0; }
.result-body { font-size: 0.9rem; }
.result-body p { margin: 0.2rem 0; line-height: 1.5; }
.result-body p:first-of-type { /* Main message */
    color: var(--text-dark); /* Use darker text for main message */
}
.result-card.real .result-body p:first-of-type { color: #065f46; } /* Darker Green */
.result-card.fake .result-body p:first-of-type { color: #991b1b; } /* Darker Red */

.result-confidence {
    font-size: 1.0rem;
    font-weight: 500;
    margin: 0.8rem 0 0.6rem 0;
    padding: 0.3rem 0.6rem;
    display: inline-block;
    border-radius: 6px;
    /* Backgrounds removed for cleaner look, color implies status */
}
.result-card.real .result-confidence { color: var(--real-color); background-color: #dcfce7; }
.result-card.fake .result-confidence { color: var(--fake-color); background-color: #fee2e2; }
.result-confidence .highlight { font-weight: 700; }
.details-summary { font-size: 0.75rem; color: var(--text-light); margin-top: 0.8rem; }

/* Error Message */
.error-message {
    background: var(--fake-bg); color: var(--fake-color);
    padding: 0.8rem 1rem; border-radius: var(--input-radius);
    border: 1px solid var(--fake-border); border-left: 4px solid var(--fake-color);
    margin: 1rem 0 0.5rem 0; /* Added margin top */
    text-align: left; font-weight: 500; font-size: 0.9rem;
}

/* Footer Styling */
.footer {
    width: 100%;
    background: var(--bg-card); /* Use card background for footer */
    border-top: 1px solid var(--border-color); /* Separator line */
    padding: 2rem 1rem 1.5rem 1rem;
    margin-top: 3rem; /* Ensure space above footer */
    color: var(--text-light);
    box-sizing: border-box;
}
.footer-content {
    max-width: 1000px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem; /* Reduced gap */
    background: transparent;
}
.footer-section { background: transparent; }
.footer-section h3 {
    color: var(--text-dark);
    font-weight: 600;
    margin-bottom: 0.8rem;
    font-size: 1.0rem;
}
.footer-section p, .footer-section a {
    color: var(--text-medium);
    font-size: 0.85rem;
    line-height: 1.6;
    text-decoration: none;
    transition: color 0.2s ease;
}
.footer-section a:hover { color: var(--primary-color); }
.social-links { margin-top: 0.8rem; display: flex; gap: 0.8rem; }
.social-links a {
    color: var(--text-light); font-size: 1.1rem;
    transition: color 0.2s ease, transform 0.2s ease;
}
.social-links a:hover { color: var(--primary-color); transform: translateY(-2px); }
.footer-bottom {
    text-align: center; padding-top: 1.5rem; margin-top: 1.5rem;
    border-top: 1px solid var(--border-color);
    color: var(--text-light); font-size: 0.8rem;
    background: transparent;
}
.footer-bottom p { margin: 0; }

/* Responsive Adjustments */
@media (max-width: 900px) {
    .app-header { padding: 0.8rem 1rem; }
    .nav-links { display: none; } /* Hide nav links on smaller screens */
    .main-content-wrapper { padding: 1.5rem 0.5rem; align-items: flex-start; } /* Align top on mobile */
    .responsive-row { flex-direction: column; gap: 1.5rem; }
    .input-col, .result-col { max-width: 550px; margin-left: auto; margin-right: auto; width: 95%; }
    .intro-text h1 { font-size: 1.8rem; }
    .intro-text h2 { font-size: 1.0rem; }
    .footer-content { grid-template-columns: 1fr; text-align: center; }
    .social-links { justify-content: center; }
}

@media (max-width: 600px) {
    .app-header { padding: 0.6rem 1rem; }
    .logo { font-size: 1.2rem; }
    .input-col, .result-col { padding: 1.25rem 1rem; }
    .intro-text h1 { font-size: 1.6rem; }
    .intro-text h2 { font-size: 0.95rem; }
    .footer { padding: 1.5rem 1rem 1rem 1rem; }
}
"""

# --- Gradio UI Definition ---
with gr.Blocks(css=modern_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:

    # --- Header ---
    gr.HTML("""
    <header class="app-header">
        <div class="logo">DeepGuard <span>Audio</span></div>
        <nav class="nav-links">
            <a href="#">How it Works</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </nav>
    </header>
    """)

    # --- Main Content Area (Wrapper for Centering) ---
    with gr.Column(elem_classes="main-content-wrapper"): # This outer column helps control vertical centering

        # Optional: Intro text above the boxes
        gr.HTML("""
        <div class='intro-text'>
            <h1>AI Audio Detection</h1>
            <h2>Upload an audio file (WAV, MP3, OGG, FLAC) below. Our AI will analyze its characteristics to determine if it's likely authentic or generated by artificial intelligence.</h2>
        </div>
        """)

        # --- The Row containing Input and Result Columns ---
        with gr.Row(elem_classes="responsive-row"):

            # --- Input Column ---
            with gr.Column(elem_classes="input-col", scale=1): # Scale helps balance if needed
                gr.HTML('<div class="upload-label">Upload or Drop Audio File:</div>')
                audio_input = gr.Audio(type="filepath", label="", elem_id="audio_input")
                analyze_btn = gr.Button("Analyze Audio", elem_id="analyze_btn", variant="primary")

                with gr.Accordion("File Details", open=False, elem_id="details_accordion"):
                    # Hidden outputs used for logic/potential future use
                    label_output = gr.Label(label="Classification", visible=False)
                    confidence_output = gr.Text(label="Confidence Score", visible=False)
                    # Visible output for audio properties
                    audio_info_output = gr.JSON(label="Audio Properties")

            # --- Result Column ---
            with gr.Column(elem_classes="result-col", scale=1): # Scale helps balance if needed
                # Status indicator (dot + text) - initially empty
                status_indicator_output = gr.HTML(label="", elem_id="status_indicator")
                # Main result card - starts with placeholder
                result_output = gr.HTML("<div class='result-placeholder'>Analysis results will appear here</div>", elem_id="result_card")

    # --- Footer ---
    gr.HTML("""
        <footer class="footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>About DeepGuard Audio</h3>
                    <p>Using advanced feature analysis (MFCC, Chroma, Spectral Contrast) and machine learning (SVM), we aim to provide insights into audio authenticity.</p>
                    <p style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.5rem;">""" + f"Model Status: {MODEL_LOAD_STATUS}" + """</p>
                </div>
                <div class="footer-section">
                    <h3>Disclaimer</h3>
                    <p>This tool provides a likelihood estimation. Results are not definitive proof and should be interpreted with caution. Model accuracy may vary.</p>
                </div>
                <div class="footer-section">
                    <h3>Connect</h3>
                    <p><a href="#">GitHub Project</a></p>
                    <p><a href="#">Report Issue</a></p>
                     <div class="social-links">
                        <a href="#" title="Twitter / X">🐦</a>
                        <a href="#" title="LinkedIn">🔗</a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <p>© 2024 DeepGuard Initiative. For research & educational purposes.</p>
            </div>
        </footer>
    """)

    # --- Event Handlers ---
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[
            label_output,
            confidence_output,
            result_output,
            status_indicator_output,
            audio_info_output
        ],
        api_name="analyze_audio"
    )

    # Clear results when a new file is uploaded/cleared for a cleaner UX
    def clear_outputs_on_change():
         # Return default/empty values for each output that shows results
         return "N/A", "0%", "<div class='result-placeholder'>Upload an audio file and click Analyze</div>", "", {}

    audio_input.change(
         fn=clear_outputs_on_change,
         inputs=[], # No input needed for clearing
         outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output],
         # Triggering change on clear might need queue=False depending on Gradio version behavior
         # queue=False
    )

    # --- Examples ---
    example_dir = "examples"
    os.makedirs(example_dir, exist_ok=True)
    example_file_path = os.path.join(example_dir, "example_audio.wav") # Make sure this file exists
    if not os.path.exists(example_file_path):
        try:
            sr_example = 22050; duration_example = 1
            silence = np.zeros(int(sr_example * duration_example))
            sf.write(example_file_path, silence, sr_example)
            logger.info(f"Created dummy example file: {example_file_path}")
        except Exception as ex_err:
             logger.warning(f"Could not create dummy example file: {ex_err}. Examples may not load.")

    if os.path.exists(example_file_path):
        gr.Examples(
            examples=[[example_file_path]], # Provide path(s) to actual audio files
            inputs=[audio_input],
            # Outputs must match the click handler exactly
            outputs=[label_output, confidence_output, result_output, status_indicator_output, audio_info_output],
            fn=analyze_audio,
            cache_examples="lazy" # Cache results unless input changes significantly
        )

# --- Launch App ---
if __name__ == "__main__":
    demo.launch()