import os
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Load the savsted model
model_file = "emotion_model.h5"
loaded_model = tf.keras.models.load_model(model_file)
print(f"Model loaded from {model_file}")


# Define the function to extract MFCC features from audio
def extract_features(audio_path, duration=3, sr=22050, n_mfcc=13):
    # Load the audio file
    audio, sr = librosa.load(audio_path, duration=duration, sr=sr)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Take the mean of each MFCC feature (average across time)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# Function to predict emotion from an audio file
def predict_emotion(audio_path):
    # Extract features from the audio file
    features = extract_features(audio_path)

    # Reshape the features to match the model's input shape
    features = features.reshape(1, 13, 1, 1)  # Adjust the shape if necessary

    # Make prediction
    prediction = loaded_model.predict(features)

    # Get the predicted emotion label
    predicted_label = np.argmax(prediction)

    # Define the emotion classes
    label_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad'}

    # Convert the predicted label to the emotion name
    predicted_emotion = label_mapping[predicted_label]

    return predicted_emotion


# Function to play audio using soundfile
def play_audio(audio_path):
    data, samplerate = sf.read(audio_path)  # Read audio file using soundfile
    sf.write('temp.wav', data, samplerate)  # Write it to a temporary file (if needed)
    os.system('start temp.wav')  # Play the audio on Windows (use 'open' for macOS, 'xdg-open' for Linux)


# Function to handle the file selection and emotion prediction
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    if file_path:
        try:
            # Display loading message
            status_label.config(text="Predicting emotion...", fg="blue")
            status_label.update()

            # Predict emotion from the selected file
            predicted_emotion = predict_emotion(file_path)
            emotion_label.config(text=f"Predicted Emotion: {predicted_emotion}", fg="green")

            # Play the audio corresponding to the predicted emotion
            play_audio(file_path)

            # Update status
            status_label.config(text="Prediction Complete", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            status_label.config(text="Error occurred!", fg="red")


# Create the main window
window = tk.Tk()
window.title("Emotion Recognition from Audio")
window.geometry("500x350")  # Set a fixed window size
window.resizable(False, False)  # Prevent resizing

# Create a frame for the content
frame = tk.Frame(window)
frame.pack(padx=20, pady=20, fill="both", expand=True)

# Create a label to show status
status_label = tk.Label(frame, text="Select an audio file to predict emotion", font=("Arial", 14))
status_label.grid(row=0, column=0, columnspan=2, pady=20)

# Create a label to show the predicted emotion
emotion_label = tk.Label(frame, text="Predicted Emotion: None", font=("Arial", 14))
emotion_label.grid(row=1, column=0, columnspan=2, pady=10)

# Create a button to browse and select an audio file
browse_button = tk.Button(frame, text="Browse Audio File", font=("Arial", 14), command=browse_file, bg="#4CAF50",
                          fg="white", relief="flat", padx=10, pady=5)
browse_button.grid(row=2, column=0, columnspan=2, pady=20)

# Run the Tkinter event loop
window.mainloop()
