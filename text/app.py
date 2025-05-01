import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os
from datetime import datetime

# Load model and vectorizer
model = joblib.load("logistic_model_bow.pkl")
vectorizer = joblib.load("count_vectorizer_bow.pkl")

# Custom emotion-emoji mapping (adjust based on your dataset labels)
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "😊",
    "joy": "😂", "neutral": "😐", "sad": "😔", "shame": "😳", "surprise": "😮"
}

# CSV setup
csv_file = "emotion_predictions.csv"
write_header = not os.path.exists(csv_file)

# Prediction functions
def predict_emotion(text):
    vect_text = vectorizer.transform([text])
    return model.predict(vect_text)[0]

def predict_proba(text):
    vect_text = vectorizer.transform([text])
    return model.predict_proba(vect_text)

# Streamlit UI
def main():
    st.set_page_config(page_title="Text Emotion Classifier", layout="wide")
    st.sidebar.title("⚙️ Settings")
    save_pred = st.sidebar.checkbox("💾 Save Predictions")

    st.title("🧠 Text Emotion Detection App")
    st.write("Analyze the emotion behind any piece of text using a trained ML model!")

    with st.form(key="emotion_form"):
        user_input = st.text_area("Enter your text here 👇")
        submit_button = st.form_submit_button(label="Predict")

    if submit_button and user_input.strip() != "":
        col1, col2 = st.columns(2)
        prediction = predict_emotion(user_input)
        probabilities = predict_proba(user_input)
        proba_df = pd.DataFrame(probabilities, columns=model.classes_).T.reset_index()
        proba_df.columns = ["Emotion", "Probability"]

        with col1:
            st.subheader("📜 Original Text")
            st.write(user_input)

            st.subheader("🔍 Prediction")
            emoji = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"**Prediction:** {prediction} {emoji}")
            st.markdown(f"**Confidence:** `{np.max(probabilities):.2f}`")

        with col2:
            st.subheader("📊 Prediction Probabilities")
            chart = alt.Chart(proba_df).mark_bar().encode(
                x="Emotion", y="Probability", color="Emotion"
            )
            st.altair_chart(chart, use_container_width=True)

        # Save to CSV
        if save_pred:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([[timestamp, user_input, prediction, np.max(probabilities)]],
                                   columns=["Timestamp", "Text", "Prediction", "Confidence"])
            new_row.to_csv(csv_file, mode='a', index=False, header=write_header)
            write_header = False
            st.sidebar.success("✅ Prediction saved!")

if __name__ == "__main__":
    main()
