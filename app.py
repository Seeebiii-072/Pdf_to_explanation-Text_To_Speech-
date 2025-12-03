from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
from TTS.api import TTS
import torch
import os

app = Flask(__name__)

# Folder to store uploaded files and generated audio
UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load the TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

def extract_text_from_pdf(file_path):
    """Extracts text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text.strip()

@app.route("/")
def home():
    return "✅ Readify AI is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle document upload and generate speech"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text
    text = extract_text_from_pdf(file_path)
    if not text:
        return jsonify({"error": "No readable text found"}), 400

    # Generate audio
    output_path = os.path.join(AUDIO_FOLDER, file.filename.replace(".pdf", ".wav"))
    tts.tts_to_file(text=text[:1000], file_path=output_path)  # limit to 1000 chars for testing

    return jsonify({
        "message": "Audio generated successfully!",
        "audio_path": output_path
    })

@app.route("/download/<filename>")
def download_audio(filename):
    """Download generated audio"""
    path = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
