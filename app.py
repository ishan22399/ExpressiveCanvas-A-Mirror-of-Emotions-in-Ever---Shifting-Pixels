import os
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, session
from tensorflow.keras.models import load_model
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
import eventlet
from werkzeug.utils import secure_filename
from flask_session import Session
from dotenv import load_dotenv
import time

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads/')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'supersecretkey')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize SocketIO
socketio = SocketIO(app)

# Load the pre-trained emotion recognition model
model_path = os.getenv('MODEL_PATH', 'C:\\Users\\ASUS\\OneDrive\\Desktop\\New EDI\\Emotion detection\\fer_model.h5')
model = load_model(model_path)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Define a function to apply color transformation based on emotion
def apply_color_transformation(image_path, emotion_text):
    try:
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Define color transformation parameters for different emotions
        emotion_mapping = {
            "happy": {"hue_shift": 15, "saturation_scale": 1.3, "brightness_shift": 30},
            "sad": {"hue_shift": -10, "saturation_scale": 0.8, "brightness_shift": -30},
            "angry": {"hue_shift": 10, "saturation_scale": 1.5, "brightness_shift": 20},
            "surprise": {"hue_shift": 20, "saturation_scale": 1.2, "brightness_shift": 40}
        }

        # Get transformation parameters based on emotion text
        params = emotion_mapping.get(emotion_text.lower(), {"hue_shift": 0, "saturation_scale": 1.0, "brightness_shift": 0})

        # Apply color transformations
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv_image)

        # Adjust hue
        hue = (hue.astype(int) + params["hue_shift"]) % 180
        hue = hue.astype(np.uint8)

        # Adjust saturation
        saturation = cv2.multiply(saturation, params["saturation_scale"])
        saturation = np.clip(saturation, 0, 255).astype(np.uint8)

        # Adjust brightness
        value = cv2.add(value, params["brightness_shift"])
        value = np.clip(value, 0, 255).astype(np.uint8)

        transformed_hsv_image = cv2.merge([hue, saturation, value])
        transformed_image = cv2.cvtColor(transformed_hsv_image, cv2.COLOR_HSV2RGB)

        return transformed_image
    except Exception as e:
        print(f"Error in apply_color_transformation: {e}")
        return None

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            ext = os.path.splitext(file.filename)[1]
            filename = secure_filename(f"{uuid.uuid4().hex}{ext}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resize_and_save_image(file, filepath)
            session['uploaded_image'] = filename  # Store the filename in session
            return redirect(url_for('display_image', filename=filename))
    except Exception as e:
        print(f"Error uploading image: {e}")
        return redirect(request.url)

# Route for displaying the uploaded image
@app.route('/display/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename)

# Route for manipulating the image
@app.route('/manipulate/<filename>')
def manipulate_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('manipulate_image.html', filename=filename, filepath=filepath)

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# WebSocket connection handler
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# WebSocket message handler to start and stop manipulation
@socketio.on('start-manipulation')
def start_stop_manipulation():
    if session.get('stop_manipulation', False):
        session.pop('stop_manipulation', None)  # Reset stop signal
    else:
        session['stop_manipulation'] = True  # Set stop signal

# WebSocket message handler for emotion detection and image manipulation

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('time_measurements.db')
c = conn.cursor()

# Create a table to store time measurements if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS time_measurements
             (emotion_detection REAL, image_manipulation REAL, total_time REAL)''')

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()

# Modify the handle_emotion_detection function to continuously insert time measurements into the database
@socketio.on('emotion-detection')
def handle_emotion_detection(json):
    try:
        action = json.get('action')
        filename = json.get('filename')
        var = json.get('var')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if action == 'start':
            while var == 'T':
                # Start time measurement for emotion detection
                emotion_start_time = time.time()
                emotion = detect_emotion()
                emotion_end_time = time.time()
                emotion_time_taken = emotion_end_time - emotion_start_time
                print("Emotion time taken: ", emotion_time_taken)

                # Start time measurement for image manipulation
                manipulation_start_time = time.time()
                manipulated_image = apply_color_transformation(filepath, emotion)
                emit('update', {'image_src': image_to_base64(manipulated_image)})
                manipulation_end_time = time.time()
                manipulation_time_taken = manipulation_end_time - manipulation_start_time
                print("Manipulation time taken: ", manipulation_time_taken)

                eventlet.sleep(1)

                # Insert time measurements into the database
                conn = sqlite3.connect('time_measurements.db')
                c = conn.cursor()
                c.execute("INSERT INTO time_measurements (emotion_detection, image_manipulation, total_time) VALUES (?, ?, ?)",
                          (emotion_time_taken, manipulation_time_taken, 0))  # Total time is not needed here
                conn.commit()
                conn.close()

                if session.get('stop_manipulation', False):
                    break

            emit('done')
            session.pop('stop_manipulation', None)
    except Exception as e:
        print(f"Error in handle_emotion_detection: {e}")


def detect_emotion():
    try:
        cap = cv2.VideoCapture(0)
        emotion = 'neutral'
        for _ in range(30):  # Capture for 30 seconds
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray_resized = cv2.resize(roi_gray, (48, 48))
                roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)
                roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
                predicted_emotion = model.predict(roi_gray_resized)
                emotion = emotion_labels[np.argmax(predicted_emotion)]
        cap.release()
        return emotion
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return 'neutral'

def image_to_base64(image):
    try:
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in image_to_base64: {e}")
        return ''

def resize_and_save_image(file, filepath, max_size=(800, 800)):
    try:
        image = Image.open(file)
        image.thumbnail(max_size)
        image.save(filepath, format='JPEG', quality=85)
    except Exception as e:
        print(f"Error in resize_and_save_image: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
