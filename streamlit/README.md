# Emotion Analysis Dashboard

This project provides a web-based dashboard for analyzing emotions from facial images, text, and audio files. It consists of a Flask backend server and a Streamlit frontend.

## Features

- Facial emotion analysis from images
- Text emotion analysis
- Audio emotion analysis
- Modern and intuitive user interface
- Real-time analysis results

## Setup Instructions

1. Clone the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask backend server:
   ```bash
   python app.py
   ```
   The server will run on http://localhost:5000

2. In a new terminal, start the Streamlit frontend:
   ```bash
   streamlit run streamlit_app.py
   ```
   The dashboard will be available at http://localhost:8501

## Usage

1. **Facial Image Analysis**:
   - Upload a facial image (JPG, JPEG, or PNG)
   - Click "Analyze Face" to get emotion predictions

2. **Text Analysis**:
   - Enter text in the text area
   - Click "Analyze Text" to get emotion predictions

3. **Audio Analysis**:
   - Upload an audio file (WAV or MP3)
   - Click "Analyze Audio" to get emotion predictions

## Project Structure

- `app.py`: Flask backend server with API endpoints
- `streamlit_app.py`: Streamlit frontend dashboard
- `requirements.txt`: Project dependencies
- `uploads/`: Directory for storing uploaded files

## Note

The current version uses dummy responses for demonstration purposes. To use your actual emotion analysis models:

1. Uncomment and modify the model loading code in `app.py`
2. Update the prediction logic in each analysis endpoint
3. Ensure your models are properly trained and saved 