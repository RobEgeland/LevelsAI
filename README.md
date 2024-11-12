# LevelsAI
## Audio EQ Adjustment Application
This project is a C++ and Python-based application designed to capture audio output from a Windows system in real time, analyze its characteristics, and predict optimal equalizer (EQ) settings to dynamically adjust audio quality. The app uses WASAPI (Windows Audio Session API) for audio interception and TensorFlow with an LSTM-based model for machine learning. The aim is to create real-time adaptive EQ adjustments that enhance the listening experience based on audio content.

### Overview
This application aims to provide real-time EQ adjustments to audio output. By capturing audio packets from the default Windows audio output device, the app extracts features such as MFCCs, spectral centroid, bandwidth, and chroma. A machine learning model, trained on these features, predicts the optimal EQ settings (Bass, Mid, Treble) for each frame of audio. This allows dynamic audio adjustments suited to various types of audio content, such as music, speech, or environmental

### Features
Real-Time Audio Capture: Intercepts audio packets directly from the system audio output.
Feature Extraction: Extracts key audio features including MFCCs, spectral centroid, spectral bandwidth, spectral contrast, and chroma.
Dynamic EQ Adjustment: Continuously predicts and applies EQ settings for Bass, Mid, and Treble in real time.
Cross-Platform Model Deployment: Trained in Python, exported to ONNX format, and run in C++ using ONNX Runtime.