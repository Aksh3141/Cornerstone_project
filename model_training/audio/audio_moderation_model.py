import os
import sys
import warnings
import numpy as np
from scipy.io import wavfile
import scipy.signal
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress harmless SciPy metadata warnings
warnings.filterwarnings("ignore", module="scipy.io.wavfile")

DATASET_PATH = "dataset"
SAMPLE_RATE = 22050
DURATION = 5  
N_MELS = 128
INPUT_SHAPE = (N_MELS, 212, 1) # 212 frames for 5s of audio using TF's default STFT

CLASS_MAPPING = {
    'sexual_content': 0,
    'violence': 1,
    'hate_speech': 2,
    'neutral': 3
}

REVERSE_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

def extract_features(file_path):
    """Loads 16/32-bit audio using SciPy and computes Spectrogram using TensorFlow."""
    try:
        # 1. Load the file using SciPy (bypasses all librosa/soundfile errors)
        sr, audio = wavfile.read(file_path)
        
        # 2. Normalize audio to float32 between -1 and 1
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        if np.max(np.abs(audio)) > 0:
             audio /= np.max(np.abs(audio))

        # Handle stereo by averaging channels into mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 3. Resample using SciPy if the sample rate isn't 22050
        if sr != SAMPLE_RATE:
            num_samples = int(len(audio) * float(SAMPLE_RATE) / sr)
            audio = scipy.signal.resample(audio, num_samples)
        
        # Convert to TensorFlow Tensor
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            
        # 4. Pad or clip to exactly DURATION
        target_len = SAMPLE_RATE * DURATION
        audio_len = tf.shape(audio_tensor)[0]
        
        def pad():
            return tf.pad(audio_tensor, [[0, target_len - audio_len]], "CONSTANT")
        def clip():
            return audio_tensor[:target_len]
            
        audio_tensor = tf.cond(tf.less(audio_len, target_len), pad, clip)
        
        # 5. Compute Mel Spectrogram using native TensorFlow math
        stft = tf.signal.stft(audio_tensor, frame_length=2048, frame_step=512, window_fn=tf.signal.hann_window)
        spectrogram = tf.abs(stft)
        
        num_spectrogram_bins = tf.shape(spectrogram)[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=0.0,
            upper_edge_hertz=8000.0 
        )
        mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        log_mel_spectrogram = tf.transpose(log_mel_spectrogram)
        
        return log_mel_spectrogram[..., tf.newaxis].numpy()
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(dataset_path):
    features = []
    labels = []
    print("Loading data...")
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path) and folder in CLASS_MAPPING:
            label = CLASS_MAPPING[folder]
            print(f"Processing folder: '{folder}' as class {label}...")
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)
                    data = extract_features(file_path)
                    if data is not None:
                        if data.shape[1] != INPUT_SHAPE[1]:
                             data = tf.image.resize(data, (INPUT_SHAPE[0], INPUT_SHAPE[1])).numpy()
                        features.append(data)
                        labels.append(label)
    return np.array(features), np.array(labels)

# 1. Load Data
X, y = load_dataset(DATASET_PATH)
print(f"Dataset loaded. Shape: {X.shape}")

if len(X) == 0:
    print("Error: No audio files were loaded.")
    sys.exit()

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Model (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(4, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
print("\n--- Starting Training ---")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 5. Print Loss History
print("\n--- Training History ---")
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
print(f"Final Training Loss:   {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# 6. Final Evaluation
print("\n--- Final Test Evaluation ---")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# 7. Generate Classification Results (Predictions)
print("\n--- Sample Classification Results ---")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

num_samples_to_show = min(10, len(y_test))
for i in range(num_samples_to_show):
    true_label = REVERSE_MAPPING[y_test[i]]
    predicted_label = REVERSE_MAPPING[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]] * 100
    marker = "✅" if true_label == predicted_label else "❌"
    print(f"{marker} True: {true_label:<15} | Predicted: {predicted_label:<15} | Confidence: {confidence:.2f}%")

# 8. Save model
print("\nSaving model...")
model.save('audio_moderation_model.h5', include_optimizer=False)
print("Model saved as audio_moderation_model.h5")