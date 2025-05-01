from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import joblib
import numpy as np

app = FastAPI()

# Load the .pkl model (assumed to be a pipeline: Vectorizer + Classifier)
model = joblib.load("logistic_model_bow.pkl")

# Define your emotion labels (adjust according to your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.post("/predict")
async def predict(text: str = Form(...)):
    # try:
    print(model.predict_proba([text]))
        # prediction_proba = model.predict_proba([text])[0]
        # predicted_index = np.argmax(prediction_proba)
        # predicted_label = emotion_labels[predicted_index]

        # return JSONResponse(content={
        #     "prediction": predicted_label,
        #     "confidence": float(prediction_proba[predicted_index]),
        #     "class_probabilities": prediction_proba.tolist()
        # })

    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)
