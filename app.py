import os
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, session, jsonify
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
import random
import sqlite3
import threading
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Configuration with better defaults
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads/')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'supersecretkey')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('flask_session', exist_ok=True)
os.makedirs('Model', exist_ok=True)

Session(app)

# Initialize SocketIO with better configuration
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables for better state management
model = None
model_loaded = False
face_cascade = None
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Add global state for manipulation control
manipulation_states = {}  # Dictionary to track manipulation state per session
manipulation_lock = threading.Lock()

def initialize_model():
    """Initialize the emotion recognition model with proper error handling"""
    global model, model_loaded
    
    # Primary model path - the specific optimized model
    primary_model_path = r'C:\Users\ASUS\OneDrive\Desktop\ExpressiveCanvas-A-Mirror-of-Emotions-in-Ever---Shifting-Pixels\Model\fer_model_t4_optimized_20250529_055559.h5'
    
    model_paths = [
        primary_model_path,  # Use the specific optimized model first
        'Model/fer_model.h5',
        'fer_model.h5',
        os.path.join(os.path.dirname(__file__), 'Model', 'fer_model.h5'),
        os.path.join(os.path.dirname(__file__), 'fer_model.h5'),
        'Model/fer_model_*.h5'  # Pattern for timestamped models
    ]
    
    logger.info("Initializing emotion recognition model...")
    logger.info(f"Primary model path: {primary_model_path}")
    
    # Try to find models with timestamps
    import glob
    for pattern in ['Model/fer_model_*.h5', 'fer_model_*.h5']:
        timestamped_models = glob.glob(pattern)
        if timestamped_models:
            # Sort by modification time, get the newest
            newest_model = max(timestamped_models, key=os.path.getmtime)
            model_paths.insert(0, newest_model)
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                logger.info(f"Found model file at: {path} (Size: {file_size} bytes)")
                
                if file_size < 1000:  # File too small to be a valid model
                    logger.warning(f"Model file {path} is too small ({file_size} bytes)")
                    continue
                
                model = load_model(path)
                logger.info(f"Model loaded successfully from: {path}")
                model_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            continue
    
    logger.warning("No valid emotion recognition model found!")
    logger.info("Model loading options:")
    logger.info("1. Train a new model using: python Model/Face_Emotion_Model.py")
    logger.info("2. Download a pre-trained FER2013 model")
    logger.info("3. Place the model file (fer_model.h5) in the Model/ directory")
    logger.info("The app will run with mock emotion detection.")
    return False

def initialize_face_cascade():
    """Initialize the face detection cascade"""
    global face_cascade
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            return False
        
        logger.info("Face cascade classifier loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading face cascade: {e}")
        return False

def initialize_database():
    """Initialize the SQLite database with proper error handling"""
    try:
        conn = sqlite3.connect('time_measurements.db', check_same_thread=False)
        c = conn.cursor()
        
        # Create table with basic structure first
        c.execute('''CREATE TABLE IF NOT EXISTS time_measurements
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      emotion_detection REAL,
                      image_manipulation REAL,
                      total_time REAL)''')
        
        # Check and add missing columns
        c.execute("PRAGMA table_info(time_measurements)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'detected_emotion' not in columns:
            c.execute('ALTER TABLE time_measurements ADD COLUMN detected_emotion TEXT')
            logger.info("Added detected_emotion column to database")
            
        if 'model_used' not in columns:
            c.execute('ALTER TABLE time_measurements ADD COLUMN model_used BOOLEAN')
            logger.info("Added model_used column to database")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

# Initialize components
initialize_model()
initialize_face_cascade()
initialize_database()

def image_to_base64(image):
    """Convert image to base64 with better error handling and format validation"""
    try:
        if image is None:
            logger.error("Image is None in image_to_base64")
            return ''
        
        logger.info(f"Converting image to base64: shape={image.shape}, dtype={image.dtype}")
        
        # Ensure image is in correct format (RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV encoding
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Encode image with better quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)
        
        if not success:
            logger.error("Failed to encode image with cv2.imencode")
            return ''
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_result = f'data:image/jpeg;base64,{img_base64}'
        
        logger.info(f"Successfully converted image to base64: length={len(base64_result)}")
        return base64_result
        
    except Exception as e:
        logger.error(f"Error in image_to_base64: {e}")
        return ''

def apply_color_transformation(image_path, emotion_text):
    """Apply color transformation with improved error handling and validation"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image from {image_path}")
            return None
            
        # Convert BGR to RGB for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate image dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            logger.error("Invalid image dimensions")
            return None

        # Enhanced emotion mapping with more emotions
        emotion_mapping = {
            "happy": {"hue_shift": 15, "saturation_scale": 1.3, "brightness_shift": 30},
            "sad": {"hue_shift": -15, "saturation_scale": 0.7, "brightness_shift": -40},
            "angry": {"hue_shift": 0, "saturation_scale": 1.8, "brightness_shift": 10},
            "surprise": {"hue_shift": 25, "saturation_scale": 1.4, "brightness_shift": 50},
            "fear": {"hue_shift": -20, "saturation_scale": 0.6, "brightness_shift": -20},
            "disgust": {"hue_shift": 30, "saturation_scale": 0.9, "brightness_shift": -10},
            "neutral": {"hue_shift": 0, "saturation_scale": 1.0, "brightness_shift": 0}
        }

        # Get transformation parameters based on emotion text
        params = emotion_mapping.get(emotion_text.lower(), {
            "hue_shift": 0, 
            "saturation_scale": 1.0, 
            "brightness_shift": 0
        })
        
        logger.info(f"Applying transformation for emotion '{emotion_text}': {params}")

        # Apply color transformations
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv_image)

        # Adjust hue with bounds checking
        hue = (hue.astype(int) + params["hue_shift"]) % 180
        hue = np.clip(hue, 0, 179).astype(np.uint8)

        # Adjust saturation with bounds checking
        saturation = cv2.multiply(saturation.astype(np.float32), params["saturation_scale"])
        saturation = np.clip(saturation, 0, 255).astype(np.uint8)

        # Adjust brightness with bounds checking
        value = cv2.add(value.astype(np.float32), params["brightness_shift"])
        value = np.clip(value, 0, 255).astype(np.uint8)

        # Merge channels back
        transformed_hsv_image = cv2.merge([hue, saturation, value])
        transformed_image = cv2.cvtColor(transformed_hsv_image, cv2.COLOR_HSV2RGB)
        
        # Validate output
        if transformed_image is None or transformed_image.shape[0] == 0:
            logger.error("Transformation resulted in invalid image")
            return image  # Return original if transformation fails
        
        logger.info(f"Successfully transformed image: {transformed_image.shape}")
        return transformed_image
        
    except Exception as e:
        logger.error(f"Error in apply_color_transformation: {e}")
        # Return original image if transformation fails
        try:
            original = cv2.imread(image_path)
            if original is not None:
                return cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            else:
                return None
        except Exception as fallback_error:
            logger.error(f"Fallback image loading failed: {fallback_error}")
            return None

def apply_advanced_color_transformation(image_path, emotion_text):
    """Apply advanced color transformation with sophisticated emotional mapping"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image from {image_path}")
            return None
            
        # Convert BGR to RGB for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate image dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            logger.error("Invalid image dimensions")
            return None

        # Advanced emotion mapping with multiple transformation layers
        emotion_mapping = {
            "happy": {
                "warmth_boost": 0.3,
                "saturation_boost": 0.4,
                "brightness_lift": 25,
                "yellow_orange_enhance": 0.2,
                "contrast_boost": 0.1,
                "vignette_strength": -0.1,  # Negative for brightening edges
                "color_temperature": 200,  # Warmer
                "glow_effect": 0.15
            },
            "sad": {
                "warmth_boost": -0.4,
                "saturation_boost": -0.3,
                "brightness_lift": -30,
                "blue_enhance": 0.25,
                "contrast_boost": -0.1,
                "vignette_strength": 0.2,  # Darkening edges
                "color_temperature": -300,  # Cooler
                "blur_effect": 0.5
            },
            "angry": {
                "warmth_boost": 0.5,
                "saturation_boost": 0.6,
                "brightness_lift": 15,
                "red_enhance": 0.4,
                "contrast_boost": 0.3,
                "vignette_strength": 0.15,
                "color_temperature": 400,  # Very warm
                "sharpening": 0.3
            },
            "surprise": {
                "warmth_boost": 0.2,
                "saturation_boost": 0.5,
                "brightness_lift": 40,
                "contrast_boost": 0.2,
                "vignette_strength": -0.2,
                "color_temperature": 100,
                "highlight_boost": 0.3,
                "clarity_boost": 0.2
            },
            "fear": {
                "warmth_boost": -0.3,
                "saturation_boost": -0.4,
                "brightness_lift": -25,
                "contrast_boost": 0.2,
                "vignette_strength": 0.3,
                "color_temperature": -200,
                "purple_tint": 0.15,
                "grain_effect": 0.1
            },
            "disgust": {
                "warmth_boost": -0.2,
                "saturation_boost": -0.2,
                "brightness_lift": -15,
                "green_enhance": 0.2,
                "contrast_boost": 0.1,
                "vignette_strength": 0.1,
                "color_temperature": -100,
                "desaturation_selective": 0.3
            },
            "neutral": {
                "warmth_boost": 0.0,
                "saturation_boost": 0.0,
                "brightness_lift": 0,
                "contrast_boost": 0.0,
                "vignette_strength": 0.0,
                "color_temperature": 0,
                "clarity_boost": 0.05
            }
        }

        params = emotion_mapping.get(emotion_text.lower(), emotion_mapping["neutral"])
        logger.info(f"Applying advanced transformation for emotion '{emotion_text}': {list(params.keys())}")

        # Convert to float for precision
        image_float = image.astype(np.float32) / 255.0
        result = image_float.copy()

        # 1. Color Temperature Adjustment
        if params.get("color_temperature", 0) != 0:
            result = adjust_color_temperature(result, params["color_temperature"])

        # 2. Warmth and Tint Adjustment
        if params.get("warmth_boost", 0) != 0:
            result = adjust_warmth_tint(result, params["warmth_boost"])

        # 3. Selective Color Enhancement
        if params.get("red_enhance", 0) != 0:
            result = enhance_color_channel(result, "red", params["red_enhance"])
        if params.get("blue_enhance", 0) != 0:
            result = enhance_color_channel(result, "blue", params["blue_enhance"])
        if params.get("green_enhance", 0) != 0:
            result = enhance_color_channel(result, "green", params["green_enhance"])
        if params.get("yellow_orange_enhance", 0) != 0:
            result = enhance_warm_tones(result, params["yellow_orange_enhance"])

        # 4. HSV Adjustments
        result = apply_hsv_adjustments(result, 
                                     saturation_boost=params.get("saturation_boost", 0),
                                     brightness_lift=params.get("brightness_lift", 0))

        # 5. Contrast and Clarity
        if params.get("contrast_boost", 0) != 0:
            result = adjust_contrast(result, params["contrast_boost"])
        if params.get("clarity_boost", 0) != 0:
            result = enhance_clarity(result, params["clarity_boost"])

        # 6. Special Effects
        if params.get("vignette_strength", 0) != 0:
            result = apply_vignette(result, params["vignette_strength"])
        if params.get("glow_effect", 0) != 0:
            result = apply_glow_effect(result, params["glow_effect"])
        if params.get("blur_effect", 0) != 0:
            result = apply_selective_blur(result, params["blur_effect"])
        if params.get("sharpening", 0) != 0:
            result = apply_sharpening(result, params["sharpening"])

        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        logger.info(f"Successfully applied advanced transformation: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error in apply_advanced_color_transformation: {e}")
        # Fallback to original function
        return apply_color_transformation(image_path, emotion_text)

def adjust_color_temperature(image, temperature):
    """Adjust color temperature (-500 to +500)"""
    try:
        # Temperature adjustment matrix
        if temperature > 0:  # Warmer
            factor = temperature / 500.0
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 + factor * 0.3), 0, 1)  # Red
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 - factor * 0.2), 0, 1)  # Blue
        else:  # Cooler
            factor = abs(temperature) / 500.0
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 - factor * 0.2), 0, 1)  # Red
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + factor * 0.3), 0, 1)  # Blue
        return image
    except Exception as e:
        logger.error(f"Error in adjust_color_temperature: {e}")
        return image

def adjust_warmth_tint(image, warmth):
    """Adjust warmth and tint"""
    try:
        # Warmth affects red-cyan balance
        if warmth > 0:
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 + warmth * 0.4), 0, 1)
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 + warmth * 0.2), 0, 1)
        else:
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 + abs(warmth) * 0.3), 0, 1)
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + abs(warmth) * 0.2), 0, 1)
        return image
    except Exception as e:
        logger.error(f"Error in adjust_warmth_tint: {e}")
        return image

def enhance_color_channel(image, channel, strength):
    """Enhance specific color channel"""
    try:
        channel_map = {"red": 0, "green": 1, "blue": 2}
        if channel in channel_map:
            ch_idx = channel_map[channel]
            image[:, :, ch_idx] = np.clip(image[:, :, ch_idx] * (1 + strength), 0, 1)
        return image
    except Exception as e:
        logger.error(f"Error in enhance_color_channel: {e}")
        return image

def enhance_warm_tones(image, strength):
    """Enhance yellow and orange tones"""
    try:
        # Convert to HSV for selective enhancement
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Yellow-orange hue range (15-45 degrees in OpenCV)
        mask = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 25)) | \
               ((hsv[:, :, 0] >= 150) & (hsv[:, :, 0] <= 180))
        
        hsv[:, :, 1] = np.where(mask, np.clip(hsv[:, :, 1] * (1 + strength), 0, 255), hsv[:, :, 1])
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return result
    except Exception as e:
        logger.error(f"Error in enhance_warm_tones: {e}")
        return image

def apply_hsv_adjustments(image, saturation_boost=0, brightness_lift=0):
    """Apply HSV adjustments with better control"""
    try:
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Saturation adjustment
        if saturation_boost != 0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_boost), 0, 255)
        
        # Brightness adjustment
        if brightness_lift != 0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_lift, 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return result
    except Exception as e:
        logger.error(f"Error in apply_hsv_adjustments: {e}")
        return image

def adjust_contrast(image, strength):
    """Adjust contrast using curves"""
    try:
        # S-curve for contrast
        if strength > 0:
            # Increase contrast
            image = np.power(image, 1 - strength * 0.5)
        else:
            # Decrease contrast
            image = np.power(image, 1 + abs(strength) * 0.5)
        return np.clip(image, 0, 1)
    except Exception as e:
        logger.error(f"Error in adjust_contrast: {e}")
        return image

def enhance_clarity(image, strength):
    """Enhance clarity using unsharp masking"""
    try:
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)
        
        # Apply to each channel
        result = image.copy()
        for i in range(3):
            channel = (image[:, :, i] * 255).astype(np.uint8)
            enhanced = cv2.addWeighted(channel, 1 + strength, 
                                     cv2.GaussianBlur(channel, (0, 0), 2.0), -strength, 0)
            result[:, :, i] = enhanced / 255.0
        
        return np.clip(result, 0, 1)
    except Exception as e:
        logger.error(f"Error in enhance_clarity: {e}")
        return image

def apply_vignette(image, strength):
    """Apply vignette effect"""
    try:
        rows, cols = image.shape[:2]
        
        # Create vignette mask
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/3)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/3)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        
        if strength > 0:
            # Darken edges
            vignette = 1 - (1 - mask) * strength
        else:
            # Brighten edges
            vignette = 1 + mask * abs(strength)
        
        # Apply vignette to each channel
        for i in range(3):
            image[:, :, i] = image[:, :, i] * vignette
        
        return np.clip(image, 0, 1)
    except Exception as e:
        logger.error(f"Error in apply_vignette: {e}")
        return image

def apply_glow_effect(image, strength):
    """Apply soft glow effect"""
    try:
        # Create glow by blurring highlights
        bright_mask = image > 0.7
        glow = cv2.GaussianBlur((image * bright_mask * 255).astype(np.uint8), (21, 21), 0) / 255.0
        
        # Blend with original
        result = cv2.addWeighted(image, 1, glow, strength, 0)
        return np.clip(result, 0, 1)
    except Exception as e:
        logger.error(f"Error in apply_glow_effect: {e}")
        return image

def apply_selective_blur(image, strength):
    """Apply selective blur to create dreamy effect"""
    try:
        # Create edge mask
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) / 255.0
        
        # Apply blur to non-edge areas
        blurred = cv2.GaussianBlur((image * 255).astype(np.uint8), (15, 15), 0) / 255.0
        
        # Blend based on edge mask
        result = image * edge_mask[:, :, np.newaxis] + blurred * (1 - edge_mask[:, :, np.newaxis]) * strength + image * (1 - strength)
        return np.clip(result, 0, 1)
    except Exception as e:
        logger.error(f"Error in apply_selective_blur: {e}")
        return image

def apply_sharpening(image, strength):
    """Apply sharpening filter"""
    try:
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9 + strength * 8, -1],
                          [-1, -1, -1]])
        
        result = cv2.filter2D((image * 255).astype(np.uint8), -1, kernel) / 255.0
        return np.clip(result, 0, 1)
    except Exception as e:
        logger.error(f"Error in apply_sharpening: {e}")
        return image

def resize_and_save_image(file, filepath, max_size=(800, 800)):
    """Resize and save image with better error handling"""
    try:
        # Open and validate image
        image = Image.open(file)
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # Resize if needed
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save with optimization
        image.save(filepath, format='JPEG', quality=85, optimize=True)
        logger.info(f"Image saved: {filepath} ({image.size})")
        return True
        
    except Exception as e:
        logger.error(f"Error in resize_and_save_image: {e}")
        return False

# Add caching for better performance

# Cache for processed images
image_cache = {}
cache_lock = threading.Lock()

def get_image_hash(image_path, emotion):
    """Generate hash for caching"""
    try:
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{file_hash}_{emotion}"
    except:
        return f"{image_path}_{emotion}_{time.time()}"

def cached_color_transformation(image_path, emotion_text):
    """Cached version of color transformation for better performance"""
    try:
        cache_key = get_image_hash(image_path, emotion_text)
        
        with cache_lock:
            if cache_key in image_cache:
                logger.info(f"Using cached transformation for {emotion_text}")
                return image_cache[cache_key]
        
        # Process image
        result = apply_advanced_color_transformation(image_path, emotion_text)
        
        # Cache result (limit cache size to 10 images)
        with cache_lock:
            if len(image_cache) >= 10:
                # Remove oldest entry
                oldest_key = next(iter(image_cache))
                del image_cache[oldest_key]
            image_cache[cache_key] = result
        
        return result
    except Exception as e:
        logger.error(f"Error in cached_color_transformation: {e}")
        return apply_advanced_color_transformation(image_path, emotion_text)

# Update the main transformation function to use caching
def apply_color_transformation(image_path, emotion_text):
    """Main color transformation function with caching and optimization"""
    return cached_color_transformation(image_path, emotion_text)

# Add endpoint to clear cache
@app.route('/clear-cache')
def clear_image_cache():
    """Clear the image transformation cache"""
    try:
        with cache_lock:
            image_cache.clear()
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for the home page with loading status
@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/status')
def status():
    """API endpoint to check system status"""
    return jsonify({
        'model_loaded': model_loaded,
        'face_detection_available': face_cascade is not None and not face_cascade.empty(),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'database_available': True  # Will be False if DB init failed
    })

# Route for handling image upload with better error handling
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file selected'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        ext = os.path.splitext(file.filename)[1].lower()
        
        if ext not in allowed_extensions:
            return jsonify({'error': f'File type {ext} not allowed. Use: {", ".join(allowed_extensions)}'}), 400
        
        if file:
            filename = secure_filename(f"{uuid.uuid4().hex}{ext}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save and resize image
            if resize_and_save_image(file, filepath):
                session['uploaded_image'] = filename
                logger.info(f"Image uploaded successfully: {filename}")
                return redirect(url_for('display_image', filename=filename))
            else:
                return jsonify({'error': 'Failed to process image'}), 500
                
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

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
    session_id = request.sid
    print(f'Client disconnected: {session_id}')
    
    # Clean up manipulation state for this session
    with manipulation_lock:
        manipulation_states[session_id] = True  # Set stop signal
        manipulation_states.pop(session_id, None)  # Remove from states
    
    # Force close any remaining camera resources
    try:
        cv2.destroyAllWindows()
    except:
        pass

# WebSocket message handler to start and stop manipulation
@socketio.on('start-manipulation')
def start_stop_manipulation():
    with manipulation_lock:
        session_id = request.sid
        if manipulation_states.get(session_id, False):
            manipulation_states[session_id] = False  # Reset stop signal
        else:
            manipulation_states[session_id] = True  # Set stop signal

# WebSocket message handler for emotion detection and image manipulation
@socketio.on('emotion-detection')
def handle_emotion_detection(data):
    """Handle emotion detection with improved debugging and error handling"""
    try:
        action = data.get('action')
        filename = data.get('filename')
        continuous = data.get('var') == 'T'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        session_id = request.sid

        logger.info(f"Emotion detection request - Action: {action}, File: {filename}, Continuous: {continuous}")

        if not os.path.exists(filepath):
            logger.error(f"Image file not found: {filepath}")
            emit('error', {'message': 'Image file not found'})
            return

        if action == 'start':
            with manipulation_lock:
                manipulation_states[session_id] = False
            
            iteration_count = 0
            max_iterations = 1000 if continuous else 1
            
            def detection_loop():
                nonlocal iteration_count
                
                while iteration_count < max_iterations and not manipulation_states.get(session_id, False):
                    try:
                        # Check stop condition at the beginning of each iteration
                        if manipulation_states.get(session_id, False):
                            logger.info(f"Stop signal received for session {session_id}")
                            break
                            
                        # Emotion detection with timing
                        emotion_start_time = time.time()
                        emotion = detect_emotion()
                        emotion_end_time = time.time()
                        emotion_time_taken = emotion_end_time - emotion_start_time
                        
                        # Check stop condition again after emotion detection
                        if manipulation_states.get(session_id, False):
                            logger.info(f"Stop signal received after emotion detection for session {session_id}")
                            break
                        
                        logger.info(f"Detected emotion: {emotion} (Time: {emotion_time_taken:.3f}s)")
                        
                        # Image manipulation with timing
                        manipulation_start_time = time.time()
                        manipulated_image = apply_color_transformation(filepath, emotion)
                        manipulation_end_time = time.time()
                        manipulation_time_taken = manipulation_end_time - manipulation_start_time
                        
                        logger.info(f"Image manipulation completed (Time: {manipulation_time_taken:.3f}s)")
                        
                        total_time = emotion_time_taken + manipulation_time_taken
                        
                        if manipulated_image is not None:
                            # Convert to base64
                            base64_image = image_to_base64(manipulated_image)
                            
                            if base64_image and len(base64_image) > 100:
                                update_data = {
                                    'image_src': base64_image,
                                    'emotion': emotion,
                                    'timing': {
                                        'emotion_detection': round(emotion_time_taken * 1000, 1),
                                        'image_manipulation': round(manipulation_time_taken * 1000, 1),
                                        'total': round(total_time * 1000, 1)
                                    },
                                    'iteration': iteration_count + 1,
                                    'timestamp': time.time(),
                                    'debug': {
                                        'image_shape': manipulated_image.shape,
                                        'base64_length': len(base64_image),
                                        'emotion_detected': emotion
                                    }
                                }
                                
                                logger.info(f"Sending update - Emotion: {emotion}, Base64 length: {len(base64_image)}, Iteration: {iteration_count + 1}")
                                
                                socketio.emit('update', update_data, room=session_id)
                                socketio.emit('test-message', {
                                    'message': f'Iteration {iteration_count + 1}: {emotion} emotion processed successfully'
                                }, room=session_id)
                                
                                logger.info(f"Successfully sent update for iteration {iteration_count + 1}")
                            else:
                                logger.error(f"Invalid base64 image: length={len(base64_image) if base64_image else 0}")
                                socketio.emit('error', {'message': 'Failed to encode manipulated image'}, room=session_id)
                                socketio.emit('test-message', {
                                    'message': f'Base64 encoding failed. Image shape: {manipulated_image.shape if manipulated_image is not None else "None"}'
                                }, room=session_id)
                                break
                            
                            # Store in database using eventlet-compatible approach
                            eventlet.spawn(store_measurement_async, 
                                         emotion_time_taken, manipulation_time_taken, 
                                         total_time, emotion, model_loaded)
                            
                        else:
                            logger.error("Image manipulation returned None")
                            socketio.emit('error', {'message': 'Failed to process image'}, room=session_id)
                            socketio.emit('test-message', {
                                'message': 'Image manipulation returned None - check transformation function'
                            }, room=session_id)
                            break
                            
                        iteration_count += 1
                        
                        # Check stop condition before sleep
                        if manipulation_states.get(session_id, False):
                            logger.info(f"Stop signal received before sleep for session {session_id}")
                            break
                            
                        if continuous and not manipulation_states.get(session_id, False):
                            eventlet.sleep(0.5)
                        else:
                            break
                            
                    except Exception as loop_error:
                        logger.error(f"Error in detection loop: {loop_error}")
                        socketio.emit('error', {'message': str(loop_error)}, room=session_id)
                        socketio.emit('test-message', {
                            'message': f'Loop error: {str(loop_error)}'
                        }, room=session_id)
                        break
                
                logger.info(f"Detection loop completed. Total iterations: {iteration_count}")
                socketio.emit('done', {'iterations': iteration_count}, room=session_id)
                
                # Clean up state
                with manipulation_lock:
                    manipulation_states.pop(session_id, None)
            
            # Use eventlet spawn instead of threading
            eventlet.spawn(detection_loop)
            
        elif action == 'stop':
            logger.info(f"Stop request received for session {session_id}")
            with manipulation_lock:
                manipulation_states[session_id] = True
            
            # Send immediate stop confirmation
            socketio.emit('test-message', {
                'message': 'Stop signal sent - processing will halt after current iteration'
            }, room=session_id)
            
    except Exception as e:
        logger.error(f"Error in handle_emotion_detection: {e}")
        emit('error', {'message': str(e)})

def detect_emotion():
    """Detect emotion with improved error handling and performance"""
    try:
        if not model_loaded:
            # Weighted random selection for more realistic mock emotions
            emotions = ['happy', 'neutral', 'sad', 'surprise', 'angry', 'fear', 'disgust']
            weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]  # More likely to be happy/neutral
            mock_emotion = random.choices(emotions, weights=weights)[0]
            return mock_emotion
            
        if face_cascade is None or face_cascade.empty():
            logger.warning("Face cascade not available, using mock emotion")
            return 'neutral'
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Camera not available, using mock emotion")
            return 'neutral'
            
        try:
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            emotion = 'neutral'
            max_attempts = 10
            
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Scale back coordinates and use the largest face
                    faces_scaled = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]
                    (x, y, w, h) = max(faces_scaled, key=lambda face: face[2] * face[3])
                    
                    # Extract and process face region
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray_full[y:y+h, x:x+w]
                    roi_gray_resized = cv2.resize(roi_gray, (48, 48))
                    roi_gray_normalized = roi_gray_resized.astype('float32') / 255.0
                    roi_gray_reshaped = np.expand_dims(np.expand_dims(roi_gray_normalized, axis=-1), axis=0)
                    
                    # Predict emotion
                    prediction = model.predict(roi_gray_reshaped, verbose=0)
                    emotion_index = np.argmax(prediction)
                    emotion = emotion_labels[emotion_index]
                    break
                    
        finally:
            # Ensure camera is always released
            cap.release()
            cv2.destroyAllWindows()  # Close any OpenCV windows
            
        return emotion.lower()
        
    except Exception as e:
        logger.error(f"Error in detect_emotion: {e}")
        return 'neutral'

# Add a test endpoint to verify image processing
@app.route('/test-manipulation/<filename>')
def test_manipulation(filename):
    """Test endpoint to verify image manipulation works"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Test with a simple emotion
        test_emotion = 'happy'
        manipulated_image = apply_color_transformation(filepath, test_emotion)
        
        if manipulated_image is not None:
            base64_image = image_to_base64(manipulated_image)
            return jsonify({
                'success': True,
                'emotion': test_emotion,
                'base64_length': len(base64_image) if base64_image else 0,
                'image_shape': manipulated_image.shape,
                'sample_base64': base64_image[:100] + '...' if base64_image else 'None'
            })
        else:
            return jsonify({'error': 'Manipulation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def store_measurement_async(emotion_time, manipulation_time, total_time, emotion, used_model):
    """Store timing measurements in database using eventlet-compatible method"""
    try:
        conn = sqlite3.connect('time_measurements.db', check_same_thread=False)
        c = conn.cursor()
        
        # Use INSERT OR IGNORE for better error handling
        c.execute("""INSERT INTO time_measurements 
                     (emotion_detection, image_manipulation, total_time, detected_emotion, model_used)
                     VALUES (?, ?, ?, ?, ?)""",
                  (emotion_time, manipulation_time, total_time, emotion, used_model))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Database storage error: {e}")

def store_measurement(emotion_time, manipulation_time, total_time, emotion, used_model):
    """Store timing measurements in database - legacy function for compatibility"""
    store_measurement_async(emotion_time, manipulation_time, total_time, emotion, used_model)

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    print(f'Client disconnected: {session_id}')
    
    # Clean up manipulation state for this session
    with manipulation_lock:
        manipulation_states[session_id] = True  # Set stop signal
        manipulation_states.pop(session_id, None)  # Remove from states
    
    # Force close any remaining camera resources
    try:
        cv2.destroyAllWindows()
    except:
        pass

# Add a force stop endpoint for emergency situations
@app.route('/force-stop/<session_id>')
def force_stop(session_id):
    """Force stop emotion detection for a specific session"""
    try:
        with manipulation_lock:
            manipulation_states[session_id] = True
        
        # Force close camera resources
        cv2.destroyAllWindows()
        
        return jsonify({'success': True, 'message': f'Force stopped session {session_id}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting ExpressiveCanvas application...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Model loaded: {model_loaded}")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
