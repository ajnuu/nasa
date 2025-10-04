import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import uuid # For unique filenames
import shutil 
# --- NEW DENOISING IMPORTS (Easier to install) ---
import librosa
import numpy as np
import soundfile as sf
# You MUST install this: pip install Flask librosa numpy soundfile
# -------------------------------------------------

# --- Configuration ---
app = Flask(__name__)
app.secret_key = 'super_secret_denoiser_key' 
UPLOAD_FOLDER = 'uploads'
# Allows all common audio files since librosa can handle them
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'} 

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Spectral Subtraction Denoising Core Logic (Easily Installed) ---

def ai_denoising_process(input_filepath):
    """
    *** Spectral Subtraction Denoising Function ***

    Applies a simple, effective, and easily implemented signal processing
    technique using Librosa and NumPy to remove constant background noise.
    """
    print(f"--- Running Spectral Subtraction Denoising on: {input_filepath} ---")
    
    # Define fallback path if processing fails
    base, ext = os.path.splitext(os.path.basename(input_filepath))
    denoised_filename = f"failed_denoise_{uuid.uuid4()}{ext}" 
    denoised_filepath = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)

    try:
        # 1. Load the Audio using librosa (handles various formats)
        # We target a standard sample rate for processing
        y, sr = librosa.load(input_filepath, sr=16000) 
        
        # --- Denoising Parameters ---
        N_FFT = 1024
        HOP_LENGTH = 256
        
        # 2. Compute the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # 3. Separate Magnitude and Phase
        magnitude, phase = librosa.magphase(stft)
        
        # 4. Estimate the Noise Profile (The core of Spectral Subtraction)
        # We assume the first 10% of the audio contains mostly noise.
        # This is the "AI" part, estimating what noise looks like.
        noise_frames = magnitude[:, :int(magnitude.shape[1] * 0.1)]
        noise_magnitude_avg = np.mean(noise_frames, axis=1, keepdims=True)
        
        # 5. Subtract the Noise from the entire audio magnitude
        # Ensure we don't end up with negative values (subtract, then clamp to 0)
        clean_magnitude = np.maximum(0, magnitude - noise_magnitude_avg * 1.5) # 1.5 is the over-subtraction factor
        
        # 6. Reconstruct the STFT using the cleaned magnitude and original phase
        stft_clean = clean_magnitude * phase
        
        # 7. Inverse STFT to get the time-domain audio
        y_clean = librosa.istft(stft_clean, hop_length=HOP_LENGTH)
        
        # 8. Save the Clean Audio
        denoised_filename = f"denoised_{uuid.uuid4()}_clean.wav" 
        denoised_filepath = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)
        
        # Save the processed numpy array back to a file
        sf.write(denoised_filepath, y_clean, sr)

        print(f"--- Denoising complete. Output saved to: {denoised_filepath} ---")
        return denoised_filepath

    except Exception as e:
        print(f"An error occurred during denoising: {e}")
        flash(f'Processing failed: {e}', 'error')
        # Fallback copy if any other error occurs
        shutil.copy(input_filepath, denoised_filepath)
        return denoised_filepath


def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page (using index.html)."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and initiates the denoising process."""
    if 'audioFile' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['audioFile']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        try:
            # 1. Save the noisy input file
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}" 
            noisy_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(noisy_filepath)

            # 2. Run the AI Denoising Process
            denoised_filepath = ai_denoising_process(noisy_filepath)

            # 3. Extract just the filenames to return to the frontend
            noisy_filename = os.path.basename(noisy_filepath)
            denoised_filename = os.path.basename(denoised_filepath)
            
            # 4. Redirect the user to the results page with file identifiers
            return redirect(url_for('results', 
                                    noisy_file=noisy_filename, 
                                    denoised_file=denoised_filename))
        
        except Exception as e:
            print(f"An error occurred during overall processing: {e}")
            flash(f'Error processing file: {e}. Please ensure it is a valid audio file.', 'error')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload a common audio file type (wav, mp3, flac, ogg).', 'error')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    """Renders the page showing both the original and denoised audio players."""
    noisy_file = request.args.get('noisy_file')
    denoised_file = request.args.get('denoised_file')
    
    if not noisy_file or not denoised_file:
        return redirect(url_for('index'))

    # Renders index.html again, but passes the file names to display the results section
    return render_template('index.html', 
                           noisy_file=noisy_file, 
                           denoised_file=denoised_file)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded and processed files from the server."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # To run this app: python app.py
    # Then navigate to http://127.0.0.1:5000/
    app.run(debug=True)
