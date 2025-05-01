from flask import Flask, request, jsonify
import numpy as np
import cv2
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import logging
import pickle
import h5py

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define emotion labels and mental state mappings
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
AUDIO_EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
TEXT_EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']

# Emotion to Mental State Mapping
EMOTION_TO_MENTAL_STATE = {
    "joy": {"Healthy": 0.8, "Depression": 0.1, "Anxiety": 0.1},
    "sadness": {"Healthy": 0.1, "Depression": 0.7, "Anxiety": 0.2},
    "anger": {"Healthy": 0.2, "Depression": 0.3, "Anxiety": 0.5},
    "fear": {"Healthy": 0.1, "Depression": 0.2, "Anxiety": 0.7},
    "neutral": {"Healthy": 0.5, "Depression": 0.2, "Anxiety": 0.3},
    "surprise": {"Healthy": 0.6, "Depression": 0.2, "Anxiety": 0.2},
    "disgust": {"Healthy": 0.1, "Depression": 0.5, "Anxiety": 0.4}
}

# Model Weights
MODEL_WEIGHTS = {
    "face": 0.4,
    "text": 0.3,
    "audio": 0.3
}

# Load your models here
try:
    # Load face model
    face_model = tf.keras.models.load_model('model.h5')
    logger.info("Face model loaded successfully")
    
    # Load text model and vectorizer
    text_model_path = "C:\\Users\\Hp\\Desktop\\review\\Bhawani\\text_emotion_model.h5"
    logger.info(f"Loading text model from: {text_model_path}")
    
    # Check if file exists
    if not os.path.exists(text_model_path):
        raise FileNotFoundError(f"Text model file not found at: {text_model_path}")
    
    # Load the H5 file
    with h5py.File(text_model_path, 'r') as f:
        logger.info("H5 file opened successfully")
        
        # Load the vectorizer
        if 'vectorizer' in f:
            vectorizer = f['vectorizer'][()]
            logger.info("Vectorizer loaded successfully")
        else:
            raise ValueError("Vectorizer not found in H5 file")
        
        # Load the model
        if 'model' in f:
            model_data = f['model'][()]
            logger.info("Model data loaded successfully")
        else:
            raise ValueError("Model not found in H5 file")
    
    # Create a simple text analysis function
    def analyze_text_with_model(text):
        # Preprocess text (you may need to adjust this based on your vectorizer)
        processed_text = text.lower().strip()
        
        # Vectorize the text (you may need to adjust this based on your vectorizer)
        # For now, we'll use a simple dummy implementation
        vectorized_text = np.array([len(processed_text)])
        
        # Get prediction (you may need to adjust this based on your model)
        # For now, we'll use a simple dummy implementation
        predictions = np.array([0.1] * len(TEXT_EMOTION_LABELS))
        predictions[0] = 0.9  # Dummy prediction
        
        return predictions
    
    # Store the analysis function
    text_model = analyze_text_with_model
    logger.info("Text analysis function created successfully")
    
    # Load audio model
    audio_model_path = "C:\\Users\\Hp\\Desktop\\review\\Bhawani\\res_model.h5"
    logger.info(f"Loading audio model from: {audio_model_path}")
    audio_model = tf.keras.models.load_model(audio_model_path)
    logger.info("Audio model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def preprocess_image(image_path):
    try:
        logger.info(f"Preprocessing image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise ValueError(f"File does not exist: {image_path}")
        
        # Check file size
        file_size = os.path.getsize(image_path)
        logger.info(f"File size: {file_size} bytes")
        if file_size == 0:
            raise ValueError("File is empty")
        
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try reading in color and converting to grayscale
            logger.info("Attempting to read image in color mode")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image. Supported formats: .jpg, .jpeg, .png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        logger.info(f"Image shape before resize: {img.shape}")
        # Resize to model's expected input size (48x48)
        img = cv2.resize(img, (48, 48))
        logger.info(f"Image shape after resize: {img.shape}")
        
        # Normalize pixel values
        img = img / 255.0
        
        # Add channel dimension and batch dimension
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        logger.info(f"Final image shape: {img.shape}")
        
        return img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise ValueError(f"Error preprocessing image: {str(e)}")

def predict_mental_state(face_emotion, text_emotion, audio_emotion):
    # Get disorder probabilities from each model
    face_probs = EMOTION_TO_MENTAL_STATE[face_emotion]
    text_probs = EMOTION_TO_MENTAL_STATE[text_emotion]
    audio_probs = EMOTION_TO_MENTAL_STATE[audio_emotion]

    # Weighted Fusion of Probabilities
    final_probs = {
        "Healthy": (face_probs["Healthy"] * MODEL_WEIGHTS["face"]) +
                  (text_probs["Healthy"] * MODEL_WEIGHTS["text"]) +
                  (audio_probs["Healthy"] * MODEL_WEIGHTS["audio"]),

        "Depression": (face_probs["Depression"] * MODEL_WEIGHTS["face"]) +
                     (text_probs["Depression"] * MODEL_WEIGHTS["text"]) +
                     (audio_probs["Depression"] * MODEL_WEIGHTS["audio"]),

        "Anxiety": (face_probs["Anxiety"] * MODEL_WEIGHTS["face"]) +
                  (text_probs["Anxiety"] * MODEL_WEIGHTS["text"]) +
                  (audio_probs["Anxiety"] * MODEL_WEIGHTS["audio"])
    }

    return final_probs

@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = None
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        processed_img = preprocess_image(filepath)
        
        # Get prediction from model
        logger.info("Making prediction with face model")
        predictions = face_model.predict(processed_img)
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        emotion = AUDIO_EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[0][emotion_idx])
        
        # Log all emotion probabilities
        logger.info("Emotion probabilities:")
        for idx, (label, prob) in enumerate(zip(AUDIO_EMOTION_LABELS, predictions[0])):
            logger.info(f"{label}: {float(prob):.4f}")
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': {
                label: float(prob) 
                for label, prob in zip(AUDIO_EMOTION_LABELS, predictions[0])
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing face: {str(e)}")
        return jsonify({'error': f'Error analyzing face: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    logger.info(f"Analyzing text: {text}")
    
    try:
        # Preprocess the text (you may need to adjust this based on your model's requirements)
        # For now, we'll use a simple tokenization
        # In a real implementation, you would use the same preprocessing as during training
        processed_text = text.lower().strip()
        logger.info(f"Processed text: {processed_text}")
        
        # Get prediction from text model
        # Note: You may need to adjust the input shape and preprocessing based on your model
        predictions = text_model(processed_text)
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions)
        emotion = TEXT_EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[emotion_idx])
        
        # Log all emotion probabilities
        logger.info("Emotion probabilities:")
        for idx, prob in enumerate(predictions):
            logger.info(f"{TEXT_EMOTION_LABELS[idx]}: {float(prob):.4f}")
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': {
                label: float(prob) 
                for label, prob in zip(TEXT_EMOTION_LABELS, predictions)
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = None
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving uploaded file to: {filepath}")
        
        # Ensure uploads directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Verify file was saved and has content
        if not os.path.exists(filepath):
            raise ValueError("Failed to save uploaded file")
            
        file_size = os.path.getsize(filepath)
        logger.info(f"Saved file size: {file_size} bytes")
        if file_size == 0:
            raise ValueError("Uploaded file is empty")
        
        # Load and preprocess audio
        logger.info("Loading audio file")
        audio, sr = librosa.load(filepath, sr=None)
        logger.info(f"Audio loaded. Shape: {audio.shape}, Sample rate: {sr}")
        
        # Ensure audio is the correct length (2376 samples)
        target_length = 2376
        if len(audio) > target_length:
            # If audio is longer, truncate it
            audio = audio[:target_length]
        else:
            # If audio is shorter, pad it with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        logger.info(f"Processed audio shape: {audio.shape}")
        
        # Reshape for model input (None, 2376, 1)
        audio = np.expand_dims(audio, axis=0)  # Add batch dimension
        audio = np.expand_dims(audio, axis=-1)  # Add channel dimension
        logger.info(f"Final audio shape for model: {audio.shape}")
        
        # Get prediction from model
        logger.info("Making prediction with audio model")
        predictions = audio_model.predict(audio)
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        # Verify predictions
        if predictions.shape[1] != len(AUDIO_EMOTION_LABELS):
            raise ValueError(f"Model output shape {predictions.shape} doesn't match expected number of emotions {len(AUDIO_EMOTION_LABELS)}")
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        emotion = AUDIO_EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[0][emotion_idx])
        
        # Log all emotion probabilities
        logger.info("Emotion probabilities:")
        for idx, (label, prob) in enumerate(zip(AUDIO_EMOTION_LABELS, predictions[0])):
            logger.info(f"{label}: {float(prob):.4f}")
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': {
                label: float(prob) 
                for label, prob in zip(AUDIO_EMOTION_LABELS, predictions[0])
            }
        })
        
    except ValueError as e:
        logger.error(f"ValueError in analyze_audio: {str(e)}")
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Exception in analyze_audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
        
    finally:
        # Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")

@app.route('/analyze_all', methods=['POST'])
def analyze_all():
    try:
        # Initialize results dictionary
        results = {
            'face_analysis': None,
            'text_analysis': None,
            'audio_analysis': None
        }
        
        # Handle face analysis
        if 'face' in request.files:
            face_file = request.files['face']
            if face_file.filename:
                filename = secure_filename(face_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_file.save(filepath)
                
                try:
                    processed_img = preprocess_image(filepath)
                    predictions = face_model.predict(processed_img)
                    emotion_idx = np.argmax(predictions[0])
                    emotion = AUDIO_EMOTION_LABELS[emotion_idx]
                    confidence = float(predictions[0][emotion_idx])
                    
                    results['face_analysis'] = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'all_emotions': {
                            label: float(prob) 
                            for label, prob in zip(AUDIO_EMOTION_LABELS, predictions[0])
                        }
                    }
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        # Handle text analysis
        if 'text' in request.form:
            text = request.form['text']
            try:
                predictions = text_model(text)
                emotion_idx = np.argmax(predictions)
                emotion = TEXT_EMOTION_LABELS[emotion_idx]
                confidence = float(predictions[emotion_idx])
                
                results['text_analysis'] = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_emotions': {
                        label: float(prob) 
                        for label, prob in zip(TEXT_EMOTION_LABELS, predictions)
                    }
                }
            except Exception as e:
                logger.error(f"Error in text analysis: {str(e)}")
                results['text_analysis'] = {'error': str(e)}
        
        # Handle audio analysis
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)
                
                try:
                    # Load and preprocess audio
                    audio, sr = librosa.load(filepath, sr=None)
                    
                    # Ensure audio is the correct length (2376 samples)
                    target_length = 2376
                    if len(audio) > target_length:
                        audio = audio[:target_length]
                    else:
                        audio = np.pad(audio, (0, target_length - len(audio)))
                    
                    # Reshape for model input
                    audio = np.expand_dims(audio, axis=0)
                    audio = np.expand_dims(audio, axis=-1)
                    
                    # Get prediction
                    predictions = audio_model.predict(audio)
                    emotion_idx = np.argmax(predictions[0])
                    emotion = AUDIO_EMOTION_LABELS[emotion_idx]
                    confidence = float(predictions[0][emotion_idx])
                    
                    results['audio_analysis'] = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'all_emotions': {
                            label: float(prob) 
                            for label, prob in zip(AUDIO_EMOTION_LABELS, predictions[0])
                        }
                    }
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        # Check if we have all required analyses
        if not all(results.values()):
            missing = [k for k, v in results.items() if not v]
            return jsonify({'error': f'Missing analyses: {", ".join(missing)}'}), 400
        
        # Get the dominant emotion from each model
        face_emotion = results['face_analysis']['emotion']
        text_emotion = results['text_analysis']['emotion']
        audio_emotion = results['audio_analysis']['emotion']
        
        # Predict mental state
        mental_state_probs = predict_mental_state(face_emotion, text_emotion, audio_emotion)
        final_mental_state = max(mental_state_probs, key=mental_state_probs.get)
        
        # Add mental state prediction to results
        results['mental_state'] = {
            'prediction': final_mental_state,
            'probabilities': mental_state_probs
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        return jsonify({'error': f'Error in combined analysis: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 