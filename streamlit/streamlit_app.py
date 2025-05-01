import streamlit as st
import requests
import json
import os
from PIL import Image
import io
import base64

# Configure the page
st.set_page_config(page_title="Emotion Analysis", layout="wide")
st.title("Multi-Modal Emotion Analysis")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Create tabs for different analysis types
tab1, tab2, tab3, tab4 = st.tabs(["Face Analysis", "Text Analysis", "Audio Analysis", "Combined Analysis"])

# Face Analysis Tab
with tab1:
    st.header("Face Emotion Analysis")
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if st.button("Analyze Face"):
            files = {"file": uploaded_file}
            response = requests.post("http://localhost:5000/analyze_face", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.results = result
                
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Display emotion prediction
                st.subheader("Emotion Prediction")
                st.write(f"Predicted Emotion: {result['emotion']}")
                st.write(f"Confidence: {result['confidence']:.2%}")
                
                # Display emotion probabilities
                st.subheader("Emotion Probabilities")
                for emotion, prob in result['all_emotions'].items():
                    st.progress(prob, text=f"{emotion}: {prob:.2%}")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Text Analysis Tab
with tab2:
    st.header("Text Emotion Analysis")
    text_input = st.text_area("Enter text for analysis")
    
    if text_input and st.button("Analyze Text"):
        response = requests.post("http://localhost:5000/analyze_text", json={"text": text_input})
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.results = result
            
            # Display text analysis results
            st.subheader("Text Analysis Results")
            st.write(f"Predicted Emotion: {result['emotion']}")
            st.write(f"Confidence: {result['confidence']:.2%}")
            
            # Display emotion probabilities
            st.subheader("Emotion Probabilities")
            for emotion, prob in result['all_emotions'].items():
                st.progress(prob, text=f"{emotion}: {prob:.2%}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Audio Analysis Tab
with tab3:
    st.header("Audio Emotion Analysis")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if audio_file is not None:
        if st.button("Analyze Audio"):
            files = {"file": audio_file}
            response = requests.post("http://localhost:5000/analyze_audio", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.results = result
                
                # Display audio analysis results
                st.subheader("Audio Analysis Results")
                st.write(f"Predicted Emotion: {result['emotion']}")
                st.write(f"Confidence: {result['confidence']:.2%}")
                
                # Display emotion probabilities
                st.subheader("Emotion Probabilities")
                for emotion, prob in result['all_emotions'].items():
                    st.progress(prob, text=f"{emotion}: {prob:.2%}")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Combined Analysis Tab
with tab4:
    st.header("Combined Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Face")
        face_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="face_combined")
    
    with col2:
        st.subheader("Text")
        text_input = st.text_area("Enter text", key="text_combined")
    
    with col3:
        st.subheader("Audio")
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"], key="audio_combined")
    
    if st.button("Analyze All"):
        if not (face_file and text_input and audio_file):
            st.error("Please provide all three inputs for combined analysis")
        else:
            # Prepare the request
            files = {
                "face": face_file,
                "audio": audio_file
            }
            data = {
                "text": text_input
            }
            
            # Send request to combined analysis endpoint
            response = requests.post("http://localhost:5000/analyze_all", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.results = result
                
                # Display individual results
                st.subheader("Individual Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Face Analysis")
                    st.write(f"Emotion: {result['face_analysis']['emotion']}")
                    st.write(f"Confidence: {result['face_analysis']['confidence']:.2%}")
                
                with col2:
                    st.write("Text Analysis")
                    st.write(f"Emotion: {result['text_analysis']['emotion']}")
                    st.write(f"Confidence: {result['text_analysis']['confidence']:.2%}")
                
                with col3:
                    st.write("Audio Analysis")
                    st.write(f"Emotion: {result['audio_analysis']['emotion']}")
                    st.write(f"Confidence: {result['audio_analysis']['confidence']:.2%}")
                
                # Display mental state prediction
                st.subheader("Mental State Prediction")
                mental_state = result['mental_state']
                st.write(f"Predicted Mental State: {mental_state['prediction']}")
                
                # Display mental state probabilities
                st.subheader("Mental State Probabilities")
                for state, prob in mental_state['probabilities'].items():
                    st.progress(prob, text=f"{state}: {prob:.2%}")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Add a footer
st.markdown("---")
st.markdown("### About")
st.markdown("This application uses machine learning models to analyze emotions from face images, text, and audio.") 