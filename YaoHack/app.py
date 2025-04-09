# File: app.py 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  

from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
from posture_analyzer import PostureAnalyzer
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
import google.generativeai as genai
from threading import Lock
import tempfile
import wave
import subprocess
from moviepy.editor import AudioFileClip
import librosa
import datetime
import soundfile as sf
import json

from threading import Lock

camera_lock = Lock()

app = Flask(__name__)

# Initialize camera
def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows: CAP_DSHOW is important
    if not cap.isOpened():
        print("Camera not opened. Trying next index...")
        cap = cv2.VideoCapture(1)  # Try another camera index
        if not cap.isOpened():
            print("No available camera found.")
            return None
    print("Camera initialized successfully.")
    return cap

camera = initialize_camera()

# Initialize posture analyzer
posture_analyzer = PostureAnalyzer()

# Global detection result state
detection_result = {
    "hand": "N/A",
    "pose": "N/A",
    "score": 0,
    "trend": "stable",
    "details": {},
    "transcript": "",
    "speech_feedback": "",
    "feedback": "",  # posture feedback
    "wpm": 0,
    "filler_word_count": 0,
    "mumbled": False,
    "duration_seconds": 0
}

# Google Speech setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\User\Downloads\try it\KitaHack25.json"
speech_client = speech.SpeechClient()

# Gemini AI setup
genai.configure(api_key='AIzaSyCDjJCuN6GtZ1m0hMJURFYgXlxo3wsxLfQ')  # Replace with your actual API key
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
chatbot_session = gemini_model.start_chat(history=[])
chatbot_lock = Lock()

# Speech analysis constants
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically"]

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.bool_):  # ✅ only this, no more deprecated alias
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

import numpy as np

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def generate_frames():
    global detection_result
    if camera is None:
        print("No camera detected at startup.")
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA NOT AVAILABLE", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success:
            print("Failed to read frame from camera.")
            continue

        frame = cv2.flip(frame, 1)

        # Analyze posture safely
        frame, posture_metrics = posture_analyzer.analyze_frame(frame)
        detection_result.update(posture_metrics)
        detection_result["feedback"] = generate_feedback(posture_metrics)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === 🔍 SPEECH ANALYSIS FUNCTIONS ===
def convert_webm_to_wav(webm_path, wav_path):
    try:
        cmd = [
            'ffmpeg',
            '-i', webm_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono channel
            '-y',            # overwrite output file if exists
            wav_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Successfully converted {webm_path} to {wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        print(f"Error output: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise

def detect_filler_words(transcript):
    return sum(transcript.lower().split().count(word) for word in FILLER_WORDS)

def detect_mumbling(audio_path):
    y, sr_rate = librosa.load(audio_path)
    volume = librosa.feature.rms(y=y).mean()
    clarity = librosa.feature.spectral_centroid(y=y, sr=sr_rate).mean()
    print(f"📊 Volume: {volume:.4f}, Clarity: {clarity:.2f}")
    return volume < 0.01 or clarity < 1500

def analyze_speech(audio_path, transcript=""):
    """Analyze speech for metrics like WPM, filler words, and clarity"""
    # If transcript is empty, use Google Speech API to get it
    if not transcript:
        with open(audio_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )

        response = speech_client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

    word_count = len(transcript.split())
    y, sr_rate = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr_rate)
    wpm = (word_count / duration) * 60 if duration > 0 else 0

    filler_count = detect_filler_words(transcript)
    mumbled = detect_mumbling(audio_path)

    feedback = []
    if filler_count > 3:
        feedback.append("⚠️ Too many filler words!")
    if mumbled:
        feedback.append("⚠️ Speech was unclear or mumbled.")
    if wpm > 160:
        feedback.append("⚠️ Speaking too fast! Try slowing down.")
    elif wpm < 100:
        feedback.append("⚠️ Speaking too slow! Try a more natural pace.")
    if not feedback:
        feedback.append("✅ Speech sounded confident and clear!")

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcript": transcript.strip(),
        "duration_seconds": round(duration, 2),
        "word_count": word_count,
        "wpm": round(wpm, 2),
        "filler_word_count": filler_count,
        "mumbled": bool(mumbled),  # <-- FIX here
        "feedback": feedback
    }


    # Save to file
    save_feedback(result)
    return result

# === 💾 SAVE HISTORY ===
def save_feedback(result):
    filename = "feedback_history.json"
    history = []

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []

    history.append(result)

    with open(filename, "w") as f:
        json.dump(history, f, indent=4)

    print("\n💾 Saved feedback to feedback_history.json")

def analyze_speech_from_webm(audio_file_path):
    """Analyze speech from WebM format (web recording)"""
    # First convert WebM to WAV for analysis
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    convert_webm_to_wav(audio_file_path, temp_wav)

    
    # Now analyze the speech
    with open(temp_wav, 'rb') as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="default",
        use_enhanced=True
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)
    transcript = ""
    if response.results:
        transcript = response.results[0].alternatives[0].transcript
    
    # Get detailed speech analysis
    speech_analysis = analyze_speech(temp_wav, transcript)
    
    # Clean up temp file
    try:
        os.unlink(temp_wav)
    except:
        pass
        
    return transcript, speech_analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

@app.route('/ai_coach')
def ai_coach():
    return render_template('ai_coach.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_result')
def get_detection_result():
    return jsonify(sanitize_for_json(detection_result))

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Debug request information
    print(f"📝 Received audio file: {audio_file.filename}")
    print(f"📝 Content type: {audio_file.content_type}")
    
    try:
        # Save directly to a non-temporary file for debugging
        debug_file_path = "recorded_audio.webm"
        audio_file.save(debug_file_path)
        print(f"✅ Audio file saved to: {debug_file_path}")
        
        # Check file
        if not os.path.exists(debug_file_path):
            return jsonify({"error": "Failed to save audio file"}), 500
            
        file_size = os.path.getsize(debug_file_path)
        print(f"📊 File size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({"error": "Audio file is empty - no audio data received"}), 400

        # Process the audio file
        transcript, speech_analysis = analyze_speech_from_webm(debug_file_path)
        feedback_text = " ".join(speech_analysis["feedback"])
        
        # Update global detection result
        detection_result.update({
            'transcript': transcript,
            'speech_feedback': feedback_text,
            'wpm': speech_analysis["wpm"],
            'filler_word_count': speech_analysis["filler_word_count"],
            'mumbled': speech_analysis["mumbled"],
            'duration_seconds': speech_analysis["duration_seconds"],
            'last_activity': time.time()
        })

        return jsonify({
            "status": "success",
            "transcript": transcript,
            "feedback": feedback_text,
            "wpm": speech_analysis["wpm"],
            "filler_count": speech_analysis["filler_word_count"],
            "duration": speech_analysis["duration_seconds"]
        })
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chatbot_message', methods=['POST'])
def chatbot_message():
    message = request.json.get('message', '')
    try:
        response = chatbot_session.send_message(message)
        return jsonify({"response": response.text, "status": "success"})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "status": "error"})

def generate_feedback(metrics):
    feedback = []
    if metrics.get('hand') != "Good":
        feedback.append("Use more hand gestures to engage your audience.")
    if metrics.get('pose') != "Good":
        details = metrics.get('details', {})
        if details.get('head_forward'):
            feedback.append("Keep your head aligned with your spine.")
        if details.get('shoulders_rounded'):
            feedback.append("Roll your shoulders back to open up your posture.")
        if details.get('shoulders_uneven'):
            feedback.append("Try to keep your shoulders level.")
    return " ".join(feedback) if feedback else "Good posture! Keep it up!"

def generate_speech_feedback(transcript):
    filler_count = detect_filler_words(transcript)
    words = len(transcript.split())
    
    feedback = []
    if filler_count > 3:
        feedback.append("Try to reduce filler words like 'um' and 'uh'.")
    if len(transcript.split()) < 10:
        feedback.append("Try to expand your speech with more content.")
    if any(filler in transcript.lower() for filler in FILLER_WORDS):
        feedback.append("Reduce filler words like 'uh' and 'um'.")
    return " ".join(feedback) if feedback else "Great speaking pace and clarity!"

if __name__ == '__main__':
    app.run(debug=False, threaded=True)