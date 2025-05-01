from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the model
model = load_model("model.h5")

# Your emotion class labels (adjust if needed)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

IMG_SIZE = (48, 48)  # Required size for grayscale model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
        img = img.resize(IMG_SIZE)

        # Convert to array and shape to (1, 48, 48, 1)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)       # Shape: (1, 48, 48, 1)
        img_array = img_array / 255.0                        # Normalize to [0, 1]

        # Predict
        predictions = model.predict(img_array)
        predicted_label = emotion_labels[np.argmax(predictions)]

        return JSONResponse(content={
            "prediction": predicted_label,
            "confidence": float(np.max(predictions)),
            "class_probabilities": predictions.tolist()
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
