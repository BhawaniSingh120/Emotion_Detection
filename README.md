# Multi-Modal Emotion Recognition System

This project implements a comprehensive emotion recognition system that analyzes emotions through multiple modalities: facial expressions, speech, and text. The system combines these different approaches to provide a more robust and accurate emotion detection solution.

## Project Structure

```
project/
├── facial/          # Facial emotion recognition module
├── speech2/         # Speech emotion recognition module
├── text/            # Text emotion recognition module
├── streamlit/       # Streamlit web application
└── model4/          # Additional models and experiments
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- At least 10GB of free disk space for datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BhawaniSingh120/Emotion_Detection.git
cd Emotion_Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r streamlit/requirements.txt
```

## Dataset Setup

### Facial Emotion Recognition Dataset
1. Download the FER2013 dataset from Kaggle:
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d facial/data/
```

### Speech Emotion Recognition Datasets
1. Download the following datasets and place them in their respective directories:
   - RAVDESS: https://zenodo.org/record/1188976
   - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
   - TESS: https://tspace.library.utoronto.ca/handle/1807/24487
   - SAVEE: http://kahlan.eps.surrey.ac.uk/savee/Download.html

2. Create the following directory structure in `speech2/`:
```
speech2/
├── Ravdess/
├── Crema/
├── Tess/
└── Savee/
```

3. Extract the downloaded datasets into their respective directories.

### Text Emotion Recognition Dataset
1. Download the Emotion Dataset from Kaggle:
```bash
kaggle datasets download -d praveengovi/emotion-dataset
unzip emotion-dataset.zip -d text/data/
```

## Required Files

The following files are required for the system to function properly. Please download them from the provided sources:

### Facial Emotion Recognition
- `haarcascade_frontalface_default.xml`: Download from OpenCV's official repository
  ```bash
  wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml -O facial/haarcascade_frontalface_default.xml
  ```
- `model.h5`: Pre-trained facial emotion recognition model (will be generated after training)

### Speech Emotion Recognition
- `emotion_model.h5`: Pre-trained speech emotion recognition model (will be generated after training)
- `scaler.pkl`: Feature scaler for audio processing (will be generated after training)
- `label_encoder.pkl`: Label encoder for emotion classes (will be generated after training)

### Text Emotion Recognition
- `emotion_model.pkl`: Pre-trained text emotion recognition model (will be generated after training)
- `count_vectorizer.pkl`: Text vectorizer for feature extraction (will be generated after training)

## Training the Models

### Facial Emotion Recognition
```bash
cd facial
python train_model.py
```

### Speech Emotion Recognition
```bash
cd speech2
python train_model.py
```

### Text Emotion Recognition
```bash
cd text
python train_model.py
```

## Usage

### Streamlit Web Application
1. Navigate to the streamlit directory:
```bash
cd streamlit
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Individual Modules

#### Facial Emotion Recognition
```bash
cd facial
python main.py
```

#### Speech Emotion Recognition
```bash
cd speech2
python app.py
```

#### Text Emotion Recognition
```bash
cd text
python app.py
```

## Features

- **Facial Emotion Recognition**: Real-time emotion detection from facial expressions
- **Speech Emotion Recognition**: Emotion analysis from audio input
- **Text Emotion Recognition**: Sentiment analysis from text input
- **Multi-Modal Integration**: Combined analysis from multiple sources
- **Web Interface**: User-friendly interface for easy interaction

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for facial detection
- TensorFlow/Keras for deep learning models
- Streamlit for web interface
- Various open-source datasets used for training
