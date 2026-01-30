import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# 1. Load Data
if not os.path.exists('dataset.csv'):
    print("Error: dataset.csv not found. Run data_collector.py first.")
    exit()

print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Handle both old format (42 features) and new format (63 features)
# If dataset has 63 features, use only X, Y (first 42 features)
num_cols = len(df.columns) - 1  # Exclude label column
if num_cols == 63:
    # New format: Use only X, Y coordinates (first 42 columns)
    X = df.iloc[:, :42].values  # First 42 columns (X, Y coordinates)
    print("Using X, Y coordinates only (42 features from 63 available)")
elif num_cols == 42:
    # Old format: Use all 42 features
    X = df.iloc[:, :42].values
    print("Using 42 features (X, Y coordinates)")
else:
    raise ValueError(f"Unexpected number of features: {num_cols}. Expected 42 or 63.")

y = df.iloc[:, -1].values  # Last column (label)

# 2. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")

# 3. Split Datap
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build Model (Lightweight for Edge)
model = keras.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train
print("Training...")
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 6. Evaluation
loss, acc = model.evaluate(X_val, y_val)
print(f"Final Validation Accuracy: {acc*100:.2f}%")

# 7. Save Keras Model
model.save('model.h5')

# 8. CONVERT TO TFLITE WITH QUANTIZATION (Crucial for Raspberry Pi)
print("Converting to Quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # <--- ENABLES QUANTIZATION
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 9. Save Label Encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Done! Files 'model.tflite' and 'label_encoder.pkl' are ready for the Raspberry Pi.")