import gradio as gr
import os
import librosa
import numpy as np
import joblib
import logging
import traceback
import datetime
import tempfile
import soundfile as sf
import json

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# --- Model Loading (Ensure paths are correct) ---
MODEL_LOAD_STATUS = "Models OK"
MODEL_IN_USE = "Enhanced SVM"
try:
    model = joblib.load("enhanced_svm_model.pkl")
    scaler = joblib.load("enhanced_scaler.pkl")
    logger.info("Enhanced model and scaler loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load enhanced model/scaler: {e}. Trying fallback...")
    MODEL_LOAD_STATUS = f"WARN: Using Fallback ({e})"
    MODEL_IN_USE = "Fallback SVM"
    try:
        model = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("Fallback model loaded.")
    except Exception as e2:
        logger.error(f"FATAL: Error loading ANY model files: {e2}", exc_info=True)
        model = None
        scaler = None
        MODEL_LOAD_STATUS = f"ERROR: No Models ({e2})"
        MODEL_IN_USE = "None"

# --- Helper Functions (Improved Error Handling & Robustness) ---
def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512, max_duration_s=30):
    """Extracts features, raising specific ValueErrors for common issues."""
    try:
        # Use soundfile to get info first - more robust
        try:
            info = sf.info(audio_path)
            sr = info.samplerate
            duration = info.duration
            if duration <= 0:
                 raise ValueError("Audio file duration is zero or negative.")
            # More precise check for minimum length needed for analysis
            min_duration_for_analysis = (hop_length / sr) * 3 # Need at least ~3 frames
            if duration < min_duration_for_analysis:
                raise ValueError(f"Audio file is too short ({duration:.2f}s). Needs > {min_duration_for_analysis:.2f}s.")
        except sf.SoundFileError as sf_err:
             logger.error(f"SoundFile error reading info: {sf_err}", exc_info=True)
             # Provide a more user-friendly message for common format issues
             if "Unsupported format" in str(sf_err) or "Error opening" in str(sf_err):
                  raise ValueError(f"Unsupported audio format or corrupted file. Details: {sf_err}")
             raise ValueError(f"Could not read audio file metadata. Details: {sf_err}")
        except Exception as info_err: # Catch other info errors
            logger.error(f"Unexpected error getting audio info: {info_err}", exc_info=True)
            raise ValueError(f"Unexpected error reading audio file properties: {info_err}")


        # Load with librosa, limiting duration
        load_duration = min(duration, max_duration_s) if max_duration_s else duration
        audio_data, sr_lib = librosa.load(audio_path, sr=None, duration=load_duration) # Use sr=None to get native SR

        # Check if audio data is effectively silent (can cause NaN features)
        # Use RMS energy; threshold might need tuning
        rms_threshold = 1e-5
        if np.mean(librosa.feature.rms(y=audio_data)) < rms_threshold:
             logger.warning(f"Audio file {audio_path} appears to be silent or near-silent.")
             # Optionally raise error or allow processing (might still yield NaNs)
             # raise ValueError("Audio file appears to be silent.")

        # Check again length after load, as load might return less than expected
        if len(audio_data) < hop_length * 2: # Need enough samples for windowing
             logger.warning(f"Loaded audio data is too short from {audio_path}.")
             raise ValueError("Loaded audio content is too short for analysis.")

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr_lib, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr_lib, n_fft=n_fft, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr_lib, n_fft=n_fft, hop_length=hop_length)

        # Check for NaNs in features (can happen with silence or edge cases)
        if np.isnan(mfccs).any() or np.isnan(spectral_contrast).any() or np.isnan(chroma).any():
            logger.warning(f"NaN values detected in features for {audio_path}. Replacing NaNs with 0.")
            mfccs = np.nan_to_num(mfccs)
            spectral_contrast = np.nan_to_num(spectral_contrast)
            chroma = np.nan_to_num(chroma)
            # Optionally raise ValueError("Feature extraction failed due to invalid values (NaNs).")


        mfccs_mean = np.mean(mfccs.T, axis=0)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Final check on feature vector shape/content if needed
        combined_features = np.concatenate((mfccs_mean, spectral_contrast_mean, chroma_mean))
        if not np.isfinite(combined_features).all():
            raise ValueError("Non-finite values detected in the final feature vector.")

        return combined_features

    # Catch specific exceptions first for better messages
    except ValueError as ve: # Re-raise ValueErrors from checks above
        logger.warning(f"Validation Error during feature extraction for {audio_path}: {ve}")
        raise ve # Pass the specific message up
    except librosa.LibrosaError as lib_e:
         logger.error(f"Librosa error extracting features from {audio_path}: {lib_e}", exc_info=True)
         raise ValueError(f"Audio processing library error: {lib_e}")
    except Exception as e:
        logger.error(f"Unexpected error extracting features from {audio_path}: {e}", exc_info=True)
        # Generic message for unexpected errors
        raise ValueError("An unexpected internal error occurred during feature extraction.")


def get_audio_info(audio_path):
    """Gets audio info, handling potential errors gracefully."""
    info_dict = {"File Name": os.path.basename(audio_path) if audio_path else "N/A"}
    try:
        if not audio_path or not isinstance(audio_path, str) or not os.path.exists(audio_path):
             info_dict["Error"] = "Invalid file path provided."
             return info_dict

        # Use context manager for file size
        try:
            file_size = os.path.getsize(audio_path)
            if file_size < 1024: size_str = f"{file_size} B"
            elif file_size < 1024**2: size_str = f"{file_size / 1024:.1f} KB"
            else: size_str = f"{file_size / 1024**2:.1f} MB"
            info_dict["File Size"] = size_str
        except OSError as size_err:
             logger.warning(f"Could not get file size for {audio_path}: {size_err}")
             info_dict["File Size"] = "N/A"


        # Get metadata using soundfile
        try:
            info = sf.info(audio_path)
            info_dict.update({
                "Duration": f"{info.duration:.2f} s" if info.duration is not None else "N/A",
                "Sample Rate": f"{info.samplerate} Hz" if info.samplerate is not None else "N/A",
                "Channels": info.channels if info.channels is not None else "N/A",
                "Format": f"{info.format_info} ({info.subtype_info})" if info.format_info and info.subtype_info else "N/A",
            })
        except sf.SoundFileError as sf_err:
             logger.warning(f"SoundFile could not get metadata for {audio_path}: {sf_err}")
             info_dict["Metadata Error"] = f"Could not read details ({sf_err})"
        except Exception as meta_err:
            logger.error(f"Unexpected error reading metadata for {audio_path}: {meta_err}", exc_info=True)
            info_dict["Metadata Error"] = f"Unexpected error ({meta_err})"

        return info_dict

    except Exception as e:
        logger.error(f"General error in get_audio_info for {audio_path}: {e}", exc_info=True)
        info_dict["Error"] = f"Unexpected failure gathering info ({e})."
        return info_dict


# --- Main Analysis Function (Refined HTML Output & Error Handling) ---
def analyze_audio(audio_filepath):
    # --- Initial States ---
    initial_state_placeholder = """
    <div class='result-placeholder'>
        <svg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='1.5' stroke='currentColor'>
          <path stroke-linecap='round' stroke-linejoin='round' d='M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z' />
        </svg>
        <span>Upload an audio file and click "Analyze" to see the results.</span>
    </div>"""
    initial_audio_info = {}
    initial_label = "N/A" # Use None or specific initial state if needed
    initial_confidence = "0%"
    initial_error_html = "" # Start with no error message


    # --- Input Validation ---
    if not audio_filepath:
        # This case handles the initial state before any upload or after clearing
        return initial_label, initial_confidence, initial_state_placeholder, initial_audio_info, initial_error_html

    # --- File Exists & Permissions (Basic Check) ---
    try:
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError("Uploaded file path not found on server.")
        # Can add a read permission check if needed: os.access(audio_filepath, os.R_OK)
    except (FileNotFoundError, TypeError, OSError) as e: # Catch issues accessing the path
         error_msg = f"File Access Error: Cannot access the uploaded file '{os.path.basename(audio_filepath)}'. It might be missing or corrupted. ({e})"
         logger.error(error_msg, exc_info=True)
         error_html = f"<div class='error-message alert alert-error'><strong>Access Error:</strong> {error_msg}</div>"
         # Provide empty info, as file is inaccessible
         return initial_label, initial_confidence, initial_state_placeholder, {}, error_html


    # --- File Type Validation ---
    file_ext = os.path.splitext(audio_filepath)[1].lower().replace('.', '')
    if file_ext not in ALLOWED_EXTENSIONS:
        error_msg = f"Invalid file type (<strong>.{file_ext}</strong>). Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}."
        logger.warning(f"Invalid file type uploaded: {file_ext}")
        error_html = f"<div class='error-message alert alert-warning'><strong>Format Error:</strong> {error_msg}</div>"
        audio_info = get_audio_info(audio_filepath) # Still try to get info
        return initial_label, initial_confidence, initial_state_placeholder, audio_info, error_html


    # --- Main Analysis Block ---
    current_audio_info = get_audio_info(audio_filepath) # Get info early
    try:
        # --- Model Availability Check ---
        if model is None or scaler is None:
            error_msg = "The analysis service is currently unavailable due to a configuration issue. Please try again later."
            logger.critical("Models not loaded, cannot perform analysis.")
            # Use a more severe alert style
            error_html = f"<div class='error-message alert alert-critical'><strong>System Error:</strong> {error_msg}</div>"
            return initial_label, initial_confidence, initial_state_placeholder, current_audio_info, error_html

        logger.info(f"Starting analysis for: {current_audio_info.get('File Name', 'Unknown File')}")

        # --- Feature Extraction (Catches ValueErrors from helper) ---
        logger.info("Step 1: Extracting audio features...")
        features = extract_features(audio_filepath)

        # --- Scaling & Prediction ---
        logger.info("Step 2: Applying scaling and model prediction...")
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0] # 0 Real, 1 Fake

        # --- Confidence Calculation ---
        logger.info("Step 3: Calculating confidence score...")
        confidence = 50.0
        conf_method = "Default (50%)"
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = probabilities[prediction] * 100
                conf_method = "Probability Score"
            elif hasattr(model, 'decision_function'):
                decision_score = model.decision_function(features_scaled)[0]
                # Sigmoid scaling maps distance from boundary (0) to 50-100% confidence
                confidence = (1 / (1 + np.exp(-abs(decision_score) * 0.7))) * 100 # Slightly adjusted multiplier
                confidence = max(50.0, min(99.9, confidence)) # Clamp
                conf_method = f"Decision Boundary Distance (Score: {decision_score:.2f})"
        except Exception as conf_e:
            logger.error(f"Confidence calculation failed: {conf_e}", exc_info=True)
            conf_method = "Calculation Error"


        # --- Format SUCCESS Results ---
        label = "Real" if prediction == 0 else "Fake"
        confidence_text = f"{confidence:.1f}%"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        logger.info(f"Analysis Complete: Label={label}, Confidence={confidence_text} ({conf_method}), File={current_audio_info.get('File Name', 'Unknown File')}")

        result_class = "result-real" if label == "Real" else "result-fake"
        status_icon_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="status-icon">
          <path fill-rule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm3.857-9.809a.75.75 0 0 0-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 1 0-1.06 1.061l2.5 2.5a.75.75 0 0 0 1.137-.089l4-5.5Z" clip-rule="evenodd" />
        </svg>
        """ if label == "Real" else """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="status-icon">
          <path fill-rule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" clip-rule="evenodd" />
        </svg>
        """
        status_text = "Likely Authentic" if label == "Real" else "Potential AI Detected"
        verdict_message = "The analysis suggests the audio characteristics align with genuine human recordings." if label == "Real" else "The analysis identified patterns commonly associated with AI-generated or manipulated audio."

        # New Result Card HTML Structure
        result_html = f"""
        <div class="analysis-result-card {result_class}">
            <div class="result-main">
                <div class="result-status-badge">
                    {status_icon_svg}
                    <span>{status_text}</span>
                </div>
                <p class="result-verdict">{verdict_message}</p>
                <div class="result-confidence">
                    <span class="confidence-label">Confidence:</span>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width: {max(1, confidence):.1f}%;"></div>
                    </div>
                    <span class="confidence-value">{confidence_text}</span>
                </div>
            </div>
            <div class="result-meta">
                Analyzed: {timestamp} | Using: {MODEL_IN_USE}
            </div>
        </div>
        """

        # Return success state: label, confidence, result HTML, info dict, empty error string
        return label, confidence_text, result_html, current_audio_info, ""

    # --- Specific Error Handling during Analysis ---
    except ValueError as ve: # Catch errors from extract_features or other validation
        error_msg = f"{ve}" # Use the specific error message from the helper
        logger.warning(f"Analysis Error for {current_audio_info.get('File Name', 'Unknown File')}: {error_msg}")
        error_html = f"<div class='error-message alert alert-warning'><strong>Analysis Failed:</strong> {error_msg}</div>"
        # Return failure state but keep existing audio info
        return initial_label, initial_confidence, initial_state_placeholder, current_audio_info, error_html

    # --- Catch-all for Unexpected Errors ---
    except Exception as e:
        error_msg = "An unexpected error occurred during the analysis process. Please try again or report the issue if it persists."
        logger.critical(f"Unexpected critical error during analysis of {current_audio_info.get('File Name', 'Unknown File')}: {e}", exc_info=True)
        error_html = f"<div class='error-message alert alert-critical'><strong>System Error:</strong> {error_msg}</div>"
        # Return failure state
        return initial_label, initial_confidence, initial_state_placeholder, current_audio_info, error_html


# --- Modern Professional CSS Theme ---
modern_professional_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    /* Base Colors (Cool Blues & Grays) */
    --color-primary-50:  #eff6ff; /* Blue 50 */
    --color-primary-100: #dbeafe; /* Blue 100 */
    --color-primary-500: #3b82f6; /* Blue 500 */
    --color-primary-600: #2563eb; /* Blue 600 */
    --color-primary-700: #1d4ed8; /* Blue 700 */

    --color-neutral-50:  #f9fafb; /* Gray 50 */
    --color-neutral-100: #f3f4f6; /* Gray 100 */
    --color-neutral-200: #e5e7eb; /* Gray 200 */
    --color-neutral-300: #d1d5db; /* Gray 300 */
    --color-neutral-400: #9ca3af; /* Gray 400 */
    --color-neutral-500: #6b7280; /* Gray 500 */
    --color-neutral-600: #4b5563; /* Gray 600 */
    --color-neutral-700: #374151; /* Gray 700 */
    --color-neutral-900: #111827; /* Gray 900 */

    /* Status Colors */
    --color-success-50:  #f0fdf4;
    --color-success-100: #dcfce7;
    --color-success-500: #22c55e;
    --color-success-600: #16a34a;
    --color-success-700: #15803d;

    --color-warning-50:  #fffbeb;
    --color-warning-100: #fef3c7;
    --color-warning-500: #f59e0b;
    --color-warning-600: #d97706;
    --color-warning-700: #b45309;

    --color-danger-50:   #fef2f2;
    --color-danger-100:  #fee2e2;
    --color-danger-500:  #ef4444;
    --color-danger-600:  #dc2626;
    --color-danger-700:  #b91c1c;

    --color-critical-bg: #7f1d1d; /* Dark Red */

    --color-white: #ffffff;
    --color-black: #000000;

    /* Typography */
    --font-family-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    --font-size-base: 16px;
    --line-height-normal: 1.6;

    /* Spacing (Using 4px base) */
    --space-1: 0.25rem; /* 4px */
    --space-2: 0.5rem;  /* 8px */
    --space-3: 0.75rem; /* 12px */
    --space-4: 1rem;    /* 16px */
    --space-5: 1.25rem; /* 20px */
    --space-6: 1.5rem;  /* 24px */
    --space-8: 2rem;    /* 32px */
    --space-10: 2.5rem; /* 40px */
    --space-12: 3rem;   /* 48px */
    --space-16: 4rem;   /* 64px */

    /* Borders & Radius */
    --border-radius-sm: 0.25rem;
    --border-radius-md: 0.375rem;
    --border-radius-lg: 0.5rem;
    --border-radius-xl: 0.75rem;
    --border-radius-full: 9999px;
    --border-width: 1px;
    --border-color-default: var(--color-neutral-200);

    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);

    /* Transitions */
    --transition-duration: 150ms;
    --transition-timing: ease-in-out;
    --transition-property-common: background-color, border-color, color, fill, stroke, opacity, box-shadow, transform;
}

/* --- Base & Body --- */
*, *::before, *::after { box-sizing: border-box; }
html { font-size: var(--font-size-base); scroll-behavior: smooth; }
body, .gradio-container {
    font-family: var(--font-family-sans);
    background: linear-gradient(135deg, #f0f4ff 0%, #e6effe 100%);  /* Light blue gradient */
    color: var(--color-neutral-700);
    line-height: var(--line-height-normal);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    margin: 0;
    display: flex; 
    flex-direction: column; 
    min-height: 100vh;
}
.gradio-container { background-color: transparent !important; }

/* --- Layout Containers --- */
.app-container { width: 100%; max-width: 1200px; margin: 0 auto; padding: 0 var(--space-6); }
.app-header, .app-footer { flex-shrink: 0; }
.main-content { flex-grow: 1; padding: var(--space-10) 0; }

/* Remove Gradio Wrappers & Labels */
.gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
    background: none !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}
.gradio-container .gr-input > label, .gradio-container .gr-output > label { display: none; } /* Hide default labels */
h1, h2, h3, h4, h5, h6 { margin: 0; font-weight: 600; color: var(--color-neutral-900); line-height: 1.3; }
p { margin: 0; }
a { color: var(--color-primary-600); text-decoration: none; transition: color var(--transition-duration) var(--transition-timing); }
a:hover { color: var(--color-primary-700); }

/* --- Header --- */
.app-header {
    background: rgba(255, 255, 255, 0.95);  /* Nearly transparent white */
    backdrop-filter: blur(8px);
    padding: var(--space-4) 0;
    border-bottom: var(--border-width) solid rgba(255, 255, 255, 0.3);
    position: sticky; 
    top: 0; 
    z-index: 50;
    box-shadow: var(--shadow-sm);
}
.header-content { display: flex; justify-content: space-between; align-items: center; }
.logo-link { font-size: 1.5rem; font-weight: 700; color: var(--color-primary-600); }
.logo-link span { font-weight: 500; color: var(--color-neutral-500); margin-left: var(--space-1); }
.main-nav { display: flex; gap: var(--space-6); }
.main-nav a { font-weight: 500; color: var(--color-neutral-600); font-size: 0.95rem; }
.main-nav a:hover { color: var(--color-primary-600); }

/* --- Intro Section --- */
.intro-section { text-align: center; margin-bottom: var(--space-12); }
.intro-section h1 { font-size: 2.25rem; font-weight: 700; margin-bottom: var(--space-4); }
.intro-section h2 { font-size: 1.1rem; color: var(--color-neutral-600); font-weight: 400; max-width: 750px; margin: 0 auto; }

/* --- Main Analysis Layout --- */
.analysis-layout {
    display: grid; grid-template-columns: 1fr; /* Default to single column */
    gap: var(--space-8);
    align-items: start; /* Align tops of columns */
}
@media (min-width: 1024px) { /* Switch to two columns on large screens */
    .analysis-layout { grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); } /* Give slightly more space to results */
}

/* --- Interaction Panels (Cards) --- */
.interaction-panel {
    background: rgba(255, 255, 255, 0.9);  /* Slightly transparent white */
    backdrop-filter: blur(10px);  /* Frosted glass effect */
    border-radius: var(--border-radius-xl);
    border: var(--border-width) solid rgba(255, 255, 255, 0.8);
    box-shadow: var(--shadow-md);
    padding: var(--space-8);
    display: flex; 
    flex-direction: column;
    height: 100%; /* Ensure panels in the same row match height */
}
.panel-title {
    font-size: 1.2rem; font-weight: 600; color: var(--color-neutral-900);
    margin-bottom: var(--space-6);
    padding-bottom: var(--space-3);
    border-bottom: var(--border-width) solid var(--color-neutral-100);
}

/* --- Input Panel Specifics --- */
.input-panel .upload-label { /* Custom label above audio input */
    display: block; font-weight: 500; color: var(--color-neutral-700);
    margin-bottom: var(--space-2); font-size: 0.9rem;
}
/* Style the Gradio Audio Input Component */
.gradio-container .gr-audio {
    border: 2px dashed var(--color-neutral-300) !important;
    border-radius: var(--border-radius-lg) !important;
    background-color: var(--color-neutral-50) !important;
    padding: var(--space-6) !important;
    text-align: center;
    transition: border-color var(--transition-duration) var(--transition-timing), background-color var(--transition-duration) var(--transition-timing);
    margin-bottom: var(--space-6) !important;
}
.gradio-container .gr-audio:hover {
    border-color: var(--color-primary-500) !important;
    background-color: var(--color-primary-50) !important;
}
/* Try to style the inner dropzone text/icon */
.gradio-container .gr-audio div[data-testid="audio"] > div > div { color: var(--color-neutral-500); font-weight: 500; }
.gradio-container .gr-audio div[data-testid="audio"] svg { color: var(--color-neutral-400); margin-bottom: var(--space-2); }

/* Style the Analyze Button */
.gradio-container .gr-button-primary {
    background: var(--color-primary-600) !important;
    color: var(--color-white) !important;
    border: none !important;
    border-radius: var(--border-radius-md) !important;
    padding: var(--space-3) var(--space-6) !important;
    font-weight: 600 !important; font-size: 1rem !important;
    box-shadow: var(--shadow-sm) !important; cursor: pointer;
    width: 100%; margin-top: auto !important; /* Push button to bottom of flex container */
    transition: background-color var(--transition-duration) var(--transition-timing), transform var(--transition-duration) var(--transition-timing), box-shadow var(--transition-duration) var(--transition-timing);
}
.gradio-container .gr-button-primary:hover {
    background: var(--color-primary-700) !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md) !important;
}
.gradio-container .gr-button-primary:active { transform: translateY(-1px); box-shadow: var(--shadow-sm) !important; }

/* Style the Details Accordion */
.gradio-container .gr-accordion {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(5px);
    border: var(--border-width) solid var(--border-color-default) !important;
    border-radius: var(--border-radius-md) !important;
    margin-top: var(--space-6) !important;
    box-shadow: none !important; overflow: hidden;
}
.gradio-container .gr-accordion > button { /* Header */
    background: var(--color-neutral-50) !important;
    padding: var(--space-3) var(--space-4) !important;
    border-bottom: var(--border-width) solid var(--border-color-default) !important;
    font-weight: 500; font-size: 0.9rem; color: var(--color-neutral-600);
    text-align: left; width: 100%;
}
.gradio-container .gr-accordion > button:hover { background: var(--color-neutral-100) !important; }
.gradio-container .gr-accordion > div { /* Content */
     padding: var(--space-4) !important; background: var(--color-white) !important;
}
/* Style JSON Output within Accordion */
.gradio-container .gr-json {
    padding: var(--space-3) !important; background-color: var(--color-neutral-50) !important;
    border-radius: var(--border-radius-sm) !important; font-size: 0.8rem;
    border: var(--border-width) solid var(--color-neutral-100) !important; color: var(--color-neutral-600);
    white-space: pre-wrap; word-break: break-word; line-height: 1.5; font-family: monospace;
}
.gradio-container .gr-json pre .hljs-attr { color: var(--color-primary-700); font-weight: 500; }
.gradio-container .gr-json pre .hljs-string { color: var(--color-success-700); }
.gradio-container .gr-json pre .hljs-number { color: var(--color-warning-700); }
.gradio-container .gr-json pre .hljs-literal { color: var(--color-neutral-500); }


/* --- Output Panel Specifics --- */
.output-container { /* Wraps error and result */
    display: flex; flex-direction: column; gap: var(--space-4);
    flex-grow: 1; /* Allow placeholder to take space */
}

/* Error Message Styling */
.error-message {
    border-radius: var(--border-radius-md);
    padding: var(--space-3) var(--space-4);
    font-weight: 500; font-size: 0.9rem;
    border-width: var(--border-width) var(--border-width) var(--border-width) 4px; /* Left border emphasis */
    border-style: solid; margin: 0;
    display: flex; align-items: start; gap: var(--space-2);
}
.error-message strong { font-weight: 600; display: block; margin-bottom: var(--space-1); }
.alert { color: var(--color-neutral-700); background-color: var(--color-neutral-100); border-color: var(--color-neutral-400); }
.alert-warning { color: var(--color-warning-700); background-color: var(--color-warning-50); border-color: var(--color-warning-500); }
.alert-error { color: var(--color-danger-700); background-color: var(--color-danger-50); border-color: var(--color-danger-500); }
.alert-critical { color: var(--color-white); background-color: var(--color-critical-bg); border-color: #611616; } /* Dark red for critical */


/* Result Placeholder Styling */
.result-placeholder {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: var(--space-12) var(--space-6);
    border: 2px dashed var(--color-neutral-300); border-radius: var(--border-radius-lg);
    background-color: var(--color-neutral-100); color: var(--color-neutral-500);
    text-align: center; flex-grow: 1; min-height: 250px;
    transition: background-color var(--transition-duration) var(--transition-timing), border-color var(--transition-duration) var(--transition-timing);
}
.result-placeholder svg {
    width: var(--space-10); height: var(--space-10); margin-bottom: var(--space-4);
    stroke: var(--color-neutral-400);
}
.result-placeholder span { font-size: 1rem; font-weight: 500; }

/* Analysis Result Card Styling */
.analysis-result-card {
    border: var(--border-width) solid var(--border-color-default);
    border-left-width: 4px; /* Accent border on the left */
    border-radius: var(--border-radius-lg);
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    overflow: hidden; box-shadow: var(--shadow-sm); flex-grow: 1;
    display: flex; flex-direction: column; /* Ensure footer stays at bottom */
    transition: border-color var(--transition-duration) var(--transition-timing);
}
.result-main { padding: var(--space-6); flex-grow: 1; } /* Main content area */

.result-status-badge {
    display: inline-flex; align-items: center; gap: var(--space-2);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--border-radius-full);
    font-size: 0.85rem; font-weight: 600; margin-bottom: var(--space-4);
}
.status-icon { width: var(--space-4); height: var(--space-4); }

.result-verdict {
    font-size: 1rem; color: var(--color-neutral-700);
    margin-bottom: var(--space-6); line-height: 1.6;
}

.result-confidence {
    display: flex; align-items: center; gap: var(--space-3);
}
.confidence-label {
    font-size: 0.8rem; font-weight: 500; color: var(--color-neutral-500);
    text-transform: uppercase; letter-spacing: 0.5px; flex-shrink: 0;
}
.confidence-bar-track {
    flex-grow: 1; height: var(--space-2);
    background-color: var(--color-neutral-200);
    border-radius: var(--border-radius-full); overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: var(--border-radius-full);
    transition: width 0.5s ease-out;
}
.confidence-value {
    font-size: 1.1rem; font-weight: 700; min-width: 55px; text-align: right;
}

.result-meta {
    background-color: var(--color-neutral-50);
    padding: var(--space-3) var(--space-6);
    border-top: var(--border-width) solid var(--color-neutral-100);
    font-size: 0.75rem; color: var(--color-neutral-500); text-align: right;
    flex-shrink: 0; /* Prevent shrinking */
}

/* Result Card Color Variants */
.analysis-result-card.result-real { border-left-color: var(--color-success-500); }
.result-real .result-status-badge { 
    background-color: var(--color-success-500); /* Changed from success-50 to success-500 */
    color: var(--color-white); /* Changed to white for better contrast */
}
.result-real .confidence-bar-fill { background-color: var(--color-success-500); }
.result-real .confidence-value { color: var(--color-success-700); }

.analysis-result-card.result-fake { border-left-color: var(--color-danger-500); }
.result-fake .result-status-badge { 
    background-color: var(--color-danger-500); /* Changed from danger-50 to danger-500 */
    color: var(--color-white); /* Changed to white for better contrast */
}
.result-fake .confidence-bar-fill { background-color: var(--color-danger-500); }
.result-fake .confidence-value { color: var(--color-danger-700); }


/* --- Footer --- */
.app-footer {
    background-color: var(--color-neutral-900);
    color: var(--color-neutral-400);
    padding: var(--space-10) 0 var(--space-8) 0;
    margin-top: var(--space-16);
    font-size: 0.9rem;
}
.footer-content {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: var(--space-8);
    margin-bottom: var(--space-8);
}
.footer-section h3 {
    color: var(--color-white); font-weight: 600; margin-bottom: var(--space-4);
    font-size: 1rem;
}
.footer-section p, .footer-section a { color: var(--color-neutral-400); }
.footer-section a:hover { color: var(--color-primary-500); }
.model-status-footer { font-size: 0.8rem; color: var(--color-neutral-500); margin-top: var(--space-2); font-style: italic; }
.footer-bottom {
    text-align: center; padding-top: var(--space-6);
    border-top: 1px solid var(--color-neutral-700); /* Darker border in footer */
    color: var(--color-neutral-500); font-size: 0.85rem;
}

/* --- Responsive Adjustments --- */
@media (max-width: 1024px) { /* Stack columns on medium screens */
    .analysis-layout { grid-template-columns: 1fr; }
}
@media (max-width: 768px) { /* Adjustments for smaller tablets/mobile */
    .app-container { padding: 0 var(--space-4); }
    .main-content { padding: var(--space-8) 0; }
    .header-content { /* Maybe stack logo and nav if needed */ }
    .main-nav { display: none; } /* Hide nav on small screens */
    .intro-section h1 { font-size: 1.8rem; }
    .intro-section h2 { font-size: 1rem; }
    .interaction-panel { padding: var(--space-6); }
    .footer-content { grid-template-columns: 1fr; text-align: center; }
    .footer-section h3 { margin-top: var(--space-4); } /* Space between stacked sections */
}

/* --- Footer Styles --- */
.footer-wrapper {
    background: linear-gradient(90deg, #0369a1 0%, #0891b2 100%);  /* Blue to Cyan gradient */
    width: 100%;
    padding: 3rem 0;
    color: #fff;
    position: relative;
    z-index: 10;
    margin-top: 4rem;
}

.footer-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 3rem;
}

.footer-section {
    color: #fff;
}

.footer-section h3 {
    color: #fff;
    font-size: 1.25rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
}

.footer-section p {
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.6;
    margin-bottom: 1rem;
}

.footer-section a {
    color: rgba(255, 255, 255, 0.8);
    text-decoration: none;
    transition: color 0.3s ease;
}

.footer-section a:hover {
    color: #fff;
    text-decoration: underline;
}

.social-links {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.social-links a {
    font-size: 1.5rem;
    color: #fff;
    transition: transform 0.3s ease;
}

.social-links a:hover {
    transform: translateY(-3px);
}

.footer-bottom {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2rem 0;
    text-align: center;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 3rem;
}

.footer-bottom p {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.9rem;
}

.model-status-footer {
    margin-top: 1rem;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.6);
    font-style: italic;
}

@media (max-width: 768px) {
    .footer-content {
        grid-template-columns: 1fr;
        gap: 2rem;
        text-align: center;
    }
    
    .social-links {
        justify-content: center;
    }
}
"""

# --- Gradio UI Definition ---
with gr.Blocks(css=modern_professional_css, theme=gr.themes.Base()) as demo:

    # --- Header ---
    with gr.Row(elem_classes="app-header"):
        with gr.Column(elem_classes="app-container header-content"): # Use Column for structure inside Row
             gr.HTML(f"""
                <a href="#" class="logo-link">DeepGuard<span>Audio</span></a>
                <nav class="main-nav">
                    <a href="#how-it-works">How It Works</a>
                    <a href="#disclaimer">Disclaimer</a>
                    <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">GitHub</a>
                </nav>
            """)

    # --- Main Content Area ---
    with gr.Column(elem_classes="main-content app-container"): # Main content wrapper

        # --- Introduction Text ---
        with gr.Row(elem_classes="intro-section"):
            gr.HTML("""
                <h1>AI Audio Authenticity Analyzer</h1>
                <h2>Determine if your audio is genuine or potentially AI-generated. Upload a file below to begin the analysis using our machine learning model. Supports WAV, MP3, OGG, and FLAC formats.</h2>
            """)

        # --- Analysis Input/Output Layout ---
        with gr.Row(elem_classes="analysis-layout"):

            # --- Input Panel ---
            with gr.Column(elem_classes="interaction-panel input-panel"):
                gr.HTML('<h3 class="panel-title">1. Upload Audio File</h3>')
                gr.HTML('<label class="upload-label" for="audio-input-component">Select or Drop File:</label>')
                audio_input = gr.Audio(type="filepath", label="", elem_id="audio-input-component")

                with gr.Accordion("Show File Details", open=False, elem_id="details-accordion"):
                    audio_info_output = gr.JSON(label="", elem_id="audio-details-json")

                analyze_btn = gr.Button("Analyze Authenticity", elem_id="analyze_btn", variant="primary")

            # --- Output Panel ---
            with gr.Column(elem_classes="interaction-panel output-panel"):
                 gr.HTML('<h3 class="panel-title">2. Analysis Result</h3>')
                 with gr.Column(elem_classes="output-container"): # Container for error + result
                    # Dedicated Error Display Area
                    error_output = gr.HTML(label="", elem_id="error-display-area")

                    # Result Display Area (starts with placeholder)
                    result_output = gr.HTML(
                         value=""" <div class='result-placeholder'>
                                    <svg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='1.5' stroke='currentColor'>
                                      <path stroke-linecap='round' stroke-linejoin='round' d='M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z' />
                                    </svg>
                                    <span>Awaiting audio analysis...</span>
                                </div>""",
                         label="", elem_id="result-display-area")

    # --- Hidden State for Logic/API ---
    label_output = gr.State(value="N/A")
    confidence_output = gr.State(value="0%")

    # --- Footer ---
    with gr.Row(elem_classes="app-footer"):
        with gr.Column(elem_classes="app-container"): # Use Column for structure inside Row
             gr.HTML("""
                <div class="footer-wrapper" style="background: linear-gradient(90deg, #0369a1 0%, #0891b2 100%); width: 100%; padding: 3rem 0;">
                    <div class="footer-content">
                        <div class="footer-section" id="how-it-works">
                            <h3>How It Works</h3>
                            <p>We analyze acoustic features (MFCCs, Chroma, Spectral Contrast) using an SVM model trained to differentiate between human and AI-generated audio patterns.</p>
                            <p class="model-status-footer">Model: Advanced SVM | Status: Active</p>
                        </div>
                        <div class="footer-section" id="disclaimer">
                            <h3>Disclaimer</h3>
                            <p>This tool provides a likelihood estimate. Results are indicative, not definitive proof. Accuracy may vary with evolving AI techniques and audio quality.</p>
                        </div>
                        <div class="footer-section">
                            <h3>Contact & Support</h3>
                            <p><a href="mailto:support@deepguard.ai">support@deepguard.ai</a></p>
                            <div class="social-links">
                                <a href="#" title="Twitter">🐦</a>
                                <a href="#" title="GitHub">📚</a>
                                <a href="#" title="LinkedIn">💼</a>
                            </div>
                        </div>
                    </div>
                    <div class="footer-bottom">
                        <p>© 2025 DeepGuard Audio Initiative. All rights reserved.</p>
                    </div>
                </div>
             """)


    # --- Event Handlers ---
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[
            label_output,        # State
            confidence_output,   # State
            result_output,       # HTML result card/placeholder
            audio_info_output,   # JSON details
            error_output         # HTML error message area
        ],
        api_name="analyze_audio"
    )

    # Clear results/errors when input changes (cleared or new file uploaded)
    def clear_on_change():
        initial_placeholder = """
        <div class='result-placeholder'>
            <svg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='1.5' stroke='currentColor'>
              <path stroke-linecap='round' stroke-linejoin='round' d='M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z' />
            </svg>
            <span>Upload an audio file and click Analyze.</span>
        </div>"""
        # Reset state, result HTML, JSON info, and error HTML
        return "N/A", "0%", initial_placeholder, {}, ""

    audio_input.change(
         fn=clear_on_change,
         inputs=[], # No inputs needed for clearing
         outputs=[label_output, confidence_output, result_output, audio_info_output, error_output],
         queue=False # UI updates should be fast
    )

    # --- Examples ---
    example_dir = "examples"
    os.makedirs(example_dir, exist_ok=True) # Ensure directory exists
    # Define example file paths (replace with your actual examples)
    real_example_path = os.path.join(example_dir, "example_real.wav")
    fake_example_path = os.path.join(example_dir, "example_fake.wav")

    # Simple dummy file creation if examples don't exist
    if not os.path.exists(real_example_path):
        try: sf.write(real_example_path, np.sin(np.linspace(0, 3*440*2*np.pi, 3*22050)), 22050)
        except Exception as e: logger.warning(f"Could not create dummy real example: {e}")
    if not os.path.exists(fake_example_path):
        try: sf.write(fake_example_path, np.random.uniform(-0.3, 0.3, 4*16000), 16000)
        except Exception as e: logger.warning(f"Could not create dummy fake example: {e}")

    example_files = []
    if os.path.exists(real_example_path): example_files.append([real_example_path])
    if os.path.exists(fake_example_path): example_files.append([fake_example_path])

    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=[audio_input],
            outputs=[label_output, confidence_output, result_output, audio_info_output, error_output],
            fn=analyze_audio,
            cache_examples="lazy",
            label="Example Audio Files",
            elem_id="examples-section" # Add ID for potential styling
        )

# --- Launch App ---
if __name__ == "__main__":
    # Recommended: Enable queue for better handling of concurrent users
    # share=False for local development, True for temporary public link
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)