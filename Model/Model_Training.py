import librosa
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Directory containing the songs
songs_dir = r'C:\Users\rober\Desktop\LevelsAI\Songs'
sample_rate = 44100

# Store features and labels across all songs
all_features = []
all_labels = []

# Function to create a target EQ setting based on synthetic or simplified assumptions
def generate_eq_label(feature_vector):
    # This is a placeholder label function 
    spectral_centroid = feature_vector[0]  

    eq_label = [1, 1, 1]  # Placeholder for bass, mid, treble gains
    if spectral_centroid < 1000:
        eq_label = [1.5, 1, 0.8]  # Boost bass, keep mid, reduce treble
    elif spectral_centroid > 3000:
        eq_label = [0.8, 1, 1.2]  # Reduce bass, keep mid, boost treble
    return eq_label

# Loop through each song in the directory
for song_file in Path(songs_dir).glob("*.mp3"):  # Adjust extension if needed
    print(f"Processing {song_file.name}")

    # Load the audio file
    y, sr = librosa.load(song_file, sr=sample_rate)
    
    # Frame length and hop length
    frame_length = 2048
    hop_length = 1024

    # Extract features for each frame
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    # Combine features into a single feature vector per frame
    features = np.vstack([mfccs, spectral_contrast, chroma]).T  # Transpose to have frames as rows

    # Generate labels for each frame based on simplified rule
    labels = np.array([generate_eq_label(frame) for frame in features])

    # Append to the lists
    all_features.append(features)
    all_labels.append(labels)

# Concatenate all features and labels across songs
all_features = np.vstack(all_features)
all_labels = np.vstack(all_labels)

print("Combined dataset shape:")
print("Features:", all_features.shape)
print("Labels:", all_labels.shape)

### Step 2: Prepare Sequential Data for RNNs ###

sequence_length = 10  # Number of consecutive frames in a sequence
feature_dimension = all_features.shape[1]  # Number of features per frame

def create_sequences(features, labels, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])  # Sequence of frames
        y.append(labels[i + sequence_length - 1])  # Target EQ for the last frame in the sequence
    return np.array(X), np.array(y)

X, y = create_sequences(all_features, all_labels, sequence_length)

print("Feature sequence shape:", X.shape)  # Should be (num_sequences, sequence_length, feature_dimension)
print("Labels shape:", y.shape)  # Should be (num_sequences, label_dimension)

### Step 3: Define and Train the LSTM Model ###

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, feature_dimension), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(y.shape[1])  # Output layer matches the number of EQ bands (e.g., bass, mid, treble)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

### Step 4: Model Evaluation ###

train_score = model.evaluate(X_train, y_train, verbose=0)
val_score = model.evaluate(X_val, y_val, verbose=0)

print("Training score (MSE):", train_score)
print("Validation score (MSE):", val_score)

