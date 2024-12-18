import librosa
import numpy as np
import os
import tf2onnx
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directory containing the songs
songs_dir = r'C:\Users\rober\Desktop\LevelsAI\Training Songs'
sample_rate = 44100

# Store features and labels across all songs
all_features = []
all_labels = []

# Function to create a target EQ setting based on synthetic or simplified assumptions
# def generate_eq_label(feature_vector):
#     # Extract individual features
#     spectral_centroid = feature_vector[0]         # Brightness
#     spectral_bandwidth = feature_vector[1]        # Fullness/width of the sound
#     spectral_contrast = feature_vector[2]         # Harmonic vs. noisy
#     mfcc_1 = feature_vector[3]                    # Example MFCC coefficient (timbre)
    
#     # Initial EQ label [bass, mid, treble]
#     eq_label = [1, 1, 1]  
    
#     # Adjust bass based on spectral centroid and bandwidth
#     if spectral_centroid < 1000 and spectral_bandwidth < 100:
#         eq_label[0] = 1.5  # Boost bass if sound is dark and narrow
#     elif spectral_centroid > 3000:
#         eq_label[0] = 0.8  # Reduce bass if sound is bright

#     # Adjust mid frequencies based on spectral contrast and MFCC
#     if spectral_contrast > 20 and mfcc_1 < 0:
#         eq_label[1] = 1.2  # Boost mid if sound has high harmonic content and negative MFCC
    
#     # Adjust treble based on spectral centroid and contrast
#     if spectral_centroid > 3000 and spectral_contrast > 15:
#         eq_label[2] = 1.3  # Boost treble for bright and harmonic sounds
#     elif spectral_centroid < 1000:
#         eq_label[2] = 0.8  # Reduce treble if sound is dark

#     return eq_label

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

    # Extract spectral centroid for each frame
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    # Ensure that the extracted features are non-empty and have consistent shape
    if spectral_centroid.size > 0:
        # Transpose so that each frame is a row with one column
        features = spectral_centroid.T  # Shape: (num_frames, 1)

        # Ensure each frame has only 1 feature (spectral centroid)
        if features.shape[1] != 1:
            print(f"Inconsistent feature dimensions for {song_file.name}, skipping.")
            continue

        # Generate labels for each frame based on simplified rule
        labels = np.array([generate_eq_label([frame[0]]) for frame in features])

        # Append to lists
        all_features.append(features)
        all_labels.append(labels)
    else:
        print(f"No valid data extracted for {song_file.name}")
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

# spec = (tf.TensorSpec((None, sequence_length, feature_dimension), tf.float32),)

# # Convert the Keras model to ONNX format with the input signature
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# # Save the ONNX model
# with open("eq_model.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())
# model.save("eq_model.keras")
model.save("saved_model")

# tf.keras.saving.save_model(model, 'eq_model.keras')


print("Model saved as eq_model")

# print("Testing Model on I Will Possess Your Heart.mp3")


# def test_model_on_song(song_path, model, sequence_length=10):
#     # Load the new song
#     y, sr = librosa.load(song_path, sr=44100)
    
#     # Frame length and hop length (match with training)
#     frame_length = 2048
#     hop_length = 1024

#     # Extract features for each frame
#     # Extract features
#     mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=13)                # 13 MFCCs
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)      # 1 spectral centroid
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)    # 1 spectral bandwidth
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)      # 7 spectral contrast
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)[:10]                  # 10 chroma features

#     # Find the minimum frame count across all features
#     min_frames = min(mfccs.shape[1], spectral_centroid.shape[1], spectral_bandwidth.shape[1], spectral_contrast.shape[1], chroma.shape[1])

#     # Truncate all features to match the minimum frame count
#     mfccs = mfccs[:, :min_frames]
#     spectral_centroid = spectral_centroid[:, :min_frames]
#     spectral_bandwidth = spectral_bandwidth[:, :min_frames]
#     spectral_contrast = spectral_contrast[:, :min_frames]
#     chroma = chroma[:, :min_frames]

#     # Stack features to create the final feature matrix
#     features = np.vstack([mfccs, spectral_centroid, spectral_bandwidth, spectral_contrast, chroma]).T  # Shape: (min_frames, 32)
#     print("Final feature shape:", features.shape)  # Should print (min_frames, 32)

    

#     # Create sequences
#     num_frames = features.shape[0]
#     sequences = []
    
#     for i in range(num_frames - sequence_length):
#         sequences.append(features[i:i + sequence_length])
        
#     sequences = np.array(sequences)  # Shape: (num_sequences, sequence_length, feature_dimension)
#     sequences = np.array(sequences, dtype=np.float32)
#     # Run the model on each sequence and collect EQ predictions
#     eq_predictions = []
#     dummy_input = np.random.rand(1, 10, 32).astype(np.float32)  # Shape: (1, sequence_length, feature_dimension)
#     print("Dummy input shape:", dummy_input.shape)
#     print("Prediction for dummy input:", model.predict(dummy_input))

#     for sequence in sequences:
#         sequence = np.expand_dims(sequence, axis=0).astype(np.float32)  # Shape: (1, sequence_length, feature_dimension)
#         eq_adjustment = model.predict(sequence)[0]  # Get EQ adjustment for the sequence
#         eq_predictions.append(eq_adjustment)
    
#     return eq_predictions

# song_path = r'C:\Users\rober\Downloads\Rockstar [ ezmp3.cc ].mp3'
# eq_predictions = test_model_on_song(song_path, model)

# for i, eq in enumerate(eq_predictions[:10]):  # Print the first 10 predictions for brevity
#     print(f"Frame {i + 1} EQ Prediction: Bass: {eq[0]}, Mid: {eq[1]}, Treble: {eq[2]}")


# eq_predictions = np.array(eq_predictions)

# plt.figure(figsize=(10, 6))
# plt.plot(eq_predictions[:, 0], label="Bass")
# plt.plot(eq_predictions[:, 1], label="Mid")
# plt.plot(eq_predictions[:, 2], label="Treble")
# plt.legend()
# plt.xlabel("Frame")
# plt.ylabel("EQ Adjustment")
# plt.title("Predicted EQ Adjustments Over Time")
# plt.show()

