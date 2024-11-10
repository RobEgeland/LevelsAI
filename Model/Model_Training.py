import librosa
import numpy as np


# Load the audio file
audio_path = 'Fat_City_Reprise_-_Cool_Cat.mp3'
y, sr = librosa.load(audio_path, sr=44100)  # Load with a sample rate of 44.1kHz

# Parameters for framing
frame_length = 2048  # Frame size in samples
hop_length = 1024    # Hop length (50% overlap)

# Step 3: Extract Audio Features

# Extract Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels

# Extract MFCCs
mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=13)

# Extract Spectral Contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

# Extract Chroma Features
chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

# Combine features into a single feature vector for each frame
features = np.vstack([mfccs, spectral_contrast, chroma])

# Transpose the features to have frames as rows
features = features.T  # Shape will be (num_frames, num_features)

# Example output for feature inspection
print(f"Feature shape: {features.shape}")
print(f"Feature vector (first frame): {features[0]}")
