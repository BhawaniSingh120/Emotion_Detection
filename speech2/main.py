from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import io
import soundfile as sf

app = FastAPI()

# Load model
model = load_model("emotion_model.h5")

# Emotion classes (adjust if needed)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load audio file from UploadFile
        audio_bytes = await file.read()
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=samplerate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)  # (40,) vector

        # Reshape for model input
        input_features = np.expand_dims(mfccs, axis=0)  # Shape: (1, 40)

        # Predict
        predictions = model.predict(input_features)
        predicted_label = emotion_labels[np.argmax(predictions)]

        return JSONResponse(content={
            "prediction": predicted_label,
            "confidence": float(np.max(predictions)),
            "class_probabilities": predictions.tolist()
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
